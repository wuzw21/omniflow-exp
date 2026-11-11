"""Bound decoded RGB retention; use author's sampling schedule and resize unchanged.

Two decoder passes avoid trusting constant-FPS frame-count estimates. First counts
frames in the requested prefix; second materializes only selected RGB frames.
This changes I/O scheduling, NOT FluxMem or the vision processor. Compare against
upstream torchvision on the short prefix before using longer inputs.
"""
import av
import torch


def read_video_bounded(ele):
    from qwen_vl_utils_fluxmem.vision_process import smart_nframes
    path=ele['video']; start=float(ele.get('video_start',0)); end=float(ele['video_end'])
    def frames():
        with av.open(path) as container:
            stream=container.streams.video[0]
            for frame in container.decode(stream):
                if frame.time is None: raise RuntimeError('Video has no presentation timestamps')
                if frame.time < start: continue
                if frame.time > end: break
                yield frame
    with av.open(path) as c:
        fps=float(c.streams.video[0].average_rate)
    total=sum(1 for _ in frames())
    n=smart_nframes(ele,total_frames=total,video_fps=fps)
    if ele.get('anchor_end',False):
        step=max(fps/max(ele.get('fps',2),1e-6),1.)
        indices=((total-1)-torch.round(torch.arange(n)*step).long()).clamp(0,total-1).flip(0)
    else:
        indices=torch.linspace(0,total-1,n).round().long()
    wanted=set(indices.tolist()); selected={}; pts={}
    for i,frame in enumerate(frames()):
        if i in wanted:
            selected[i]=torch.from_numpy(frame.to_ndarray(format='rgb24')).permute(2,0,1)
            pts[i]=frame.time
    video=torch.stack([selected[i] for i in indices.tolist()])
    read_video_bounded.last_trace={'total_decoded_prefix_frames':total,'selected_indices':indices.tolist(),
                                 'selected_pts_seconds':[pts[i] for i in indices.tolist()],
                                 'cutoff_seconds':end,'source_fps':fps,'sampled_frames':n,
                                 'backend':'PyAV two passes; author smart_nframes/linspace'}
    return video,n/total*fps
