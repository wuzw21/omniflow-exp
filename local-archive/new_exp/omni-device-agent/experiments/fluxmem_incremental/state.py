"""Execution-state adapter. All decisions delegate to the original FluxMem helpers."""
import functools,sys
import torch
import torch.nn.functional as F

class Counter:
    def __init__(self,policy):
        self.counts={'match_calls':0,'match_query_tokens':0,'prune_frames':0,'merge_frames':0}
        for name,key in [('_max_sim_against_frames','match_calls'),('_apply_adjacent_pruning','prune_frames'),('_long_term_memory_merge_per_frames','merge_frames')]:
            orig=getattr(policy,name)
            def wrap(*args,_orig=orig,_key=key,**kw):
                self.counts[_key]+=len(kw['long_frames']) if _key=='merge_frames' else 1
                if _key=='match_calls':self.counts['match_query_tokens']+=kw['query_indices'].numel()
                return _orig(*args,**kw)
            setattr(policy,name,functools.wraps(orig)(wrap))


def storage_report(obj):
    """Count unique persistent tensor storage, not tensor views or allocator pools."""
    seen=set();storages={};python_bytes=0
    def walk(v):
        nonlocal python_bytes
        if id(v) in seen:return
        seen.add(id(v));python_bytes+=sys.getsizeof(v)
        if isinstance(v,torch.Tensor):
            s=v.untyped_storage();key=(str(v.device),s.data_ptr());storages[key]=s.nbytes()
        elif isinstance(v,dict):
            for k,x in v.items():walk(k);walk(x)
        elif isinstance(v,(list,tuple)):
            for x in v:walk(x)
    walk(obj)
    return {'tensor_storage_bytes':sum(storages.values()),'storages':len(storages),'python_object_bytes':python_bytes}

class IncrementalMemory:
    def __init__(self,policy,grid_hw=(8,15)):
        self.policy=policy;self.hw=grid_hw;self.reset()
    def reset(self):
        self.raw=[];self.values=None;self.norm=None;self.height=None;self.width=None;self.times=None
        self.distance=None;self.keep=None;self.frames=[];self.indices={};self.grids={};self.short=[];self.mid=[]
        self.offsets=None
    def append(self,feature):
        # Atomic input: one complete temporal block, immutable raw visual features.
        h,w=self.hw;n=h*w;assert feature.shape[0]==n
        self.raw.append(feature.detach().clone())
        i=len(self.frames);frame_id=4*i+1;start=i*n;dev=feature.device
        norm=F.normalize(feature,p=2,dim=1,eps=1e-8)
        def extend(old,new):return new.clone() if old is None else torch.cat([old,new])
        self.values=extend(self.values,feature);self.norm=extend(self.norm,norm)
        self.height=extend(self.height,torch.arange(h,device=dev).repeat_interleave(w))
        self.width=extend(self.width,torch.arange(w,device=dev).repeat(h))
        self.times=extend(self.times,torch.full((n,),frame_id,device=dev,dtype=torch.long))
        self.distance=extend(self.distance,torch.full((n,),-1.,device=dev,dtype=feature.dtype))
        self.keep=extend(self.keep,torch.zeros(n,device=dev,dtype=torch.bool))
        cur=torch.arange(start,start+n,device=dev)
        ids,indices,grids=self.policy._frame_grids(cur,self.times,self.height,self.width,self.hw)
        assert ids==[frame_id]
        self.indices.update(indices);self.grids.update(grids)
        if self.offsets is None:self.offsets=self.policy._neighbor_offsets(dev,include_center=True)
        self.frames.append(frame_id);self.short.append(frame_id)
        if i>0:
            paired,_,dist,_=self.policy._max_sim_against_frames(query_indices=cur,neighbor_frames=[self.frames[-2]],height_ids=self.height,width_ids=self.width,frame_grids=self.grids,hidden_norm=self.norm,offsets_3x3=self.offsets)
            if paired.numel()>0:self.distance[paired]=dist
        if len(self.short)>max(1,int(self.policy.short_term_frames/2)):
            evicted=self.short.pop(0)
            self.policy._apply_adjacent_pruning(frame_id=evicted,buffer_frames=self.short,frame_to_indices=self.indices,frame_grids=self.grids,height_ids=self.height,width_ids=self.width,hidden_norm=self.norm,pair_distance_cache=self.distance,keep_mask=self.keep,pair_distance_threshold=None,offsets_3x3=self.offsets)
            self.mid.append(evicted)
            if len(self.mid)>max(0,int(self.policy.mid_term_frames/2)):
                oldest=self.mid.pop(0)
                self.policy._long_term_memory_merge_per_frames(batch_index=0,long_frames={oldest},visual_indices=torch.arange(self.values.shape[0],device=dev),time_ids=self.times,height_ids=self.height,width_ids=self.width,hidden_states=self.values.unsqueeze(0),hidden_norm=self.norm,keep_mask=self.keep,grid_hw=self.hw)
    def snapshot(self):
        # Crucial: marking the current short window must NOT mutate execution state.
        keep=self.keep.clone()
        for frame in self.short:keep[self.indices[frame]]=True
        indices=keep.nonzero(as_tuple=True)[0]
        return {'indices':indices,'features':self.values[indices]}
    def raw_features(self):return torch.cat(self.raw)
    def state_report(self):
        payload={'raw':self.raw,'merged':self.values,'norm':self.norm,'height':self.height,'width':self.width,'times':self.times,'distance':self.distance,'keep':self.keep,'indices':self.indices,'grids':self.grids,'offsets':self.offsets,'frames':self.frames,'short':self.short,'mid':self.mid}
        return {**storage_report(payload),'components':{k:storage_report(v)['tensor_storage_bytes'] for k,v in payload.items()}}


def full_memory(policy,features,config,rotary,markers):
    """Run ORIGINAL process_memory_streaming, with real vision and special embeddings."""
    n,d=features.shape;h,w=8,15;blocks=n//(h*w);dev=features.device
    ids=torch.full((1,n+2),config.video_token_id,device=dev,dtype=torch.long)
    ids[0,0]=config.vision_start_token_id;ids[0,-1]=config.vision_end_token_id
    hidden=torch.cat([markers[:1],features,markers[1:]]).unsqueeze(0)
    pos=torch.zeros((3,1,n+2),device=dev,dtype=torch.long)
    pos[0,0,1:-1]=torch.arange(blocks,device=dev).repeat_interleave(h*w)*4+1
    pos[1,0,1:-1]=torch.arange(h,device=dev).repeat_interleave(w).repeat(blocks)+1
    pos[2,0,1:-1]=torch.arange(w,device=dev).repeat(h*blocks)+1
    pos[:,:, -1]=pos[:,:,1:-1].max()+1
    emb=rotary(hidden,pos)
    out=policy.process_memory_streaming(hidden_states=hidden,position_ids=pos,position_embeddings=emb,input_ids=ids,video_grid_thw=torch.tensor([[blocks,16,30]],device=dev))
    kept=out[-1][0];assert kept[0]==0 and kept[-1]==n+1
    return {'indices':kept[1:-1]-1,'features':out[0][0,1:-1]}


def compare(a,b):
    indices=torch.equal(a['indices'],b['indices'])
    equal=indices and torch.equal(a['features'],b['features'])
    err=(a['features'].float()-b['features'].float()).abs().max().item() if indices and a['features'].numel() else (0. if indices else None)
    return {'indices_equal':indices,'features_bitwise_equal':equal,'max_abs_error':err,'tokens_a':len(a['indices']),'tokens_b':len(b['indices'])}
