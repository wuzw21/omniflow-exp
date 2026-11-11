"""Causal, fixed-capacity gates. No OCR or future observations are accepted."""
from collections import OrderedDict
import re
import numpy as np
from PIL import Image

CAPACITY=4
THRESHOLDS=(.01,.025,.05,.10,.20)

def thumbnail(image):
    return np.asarray(image.convert('RGB').resize((64,64),Image.Resampling.BILINEAR),dtype=np.float32)/255

class Gate:
    def __init__(self, kind, param=0, capacity=CAPACITY):
        self.kind, self.param, self.capacity=kind,param,capacity
        self.memory=OrderedDict()
    def push(self, step, feature):
        if not self.memory or self.kind=='all':
            accept=True
        elif self.kind=='uniform':
            accept=step%int(self.param)==0
        else:
            refs=list(self.memory.values())
            if self.kind=='last': refs=refs[-1:]
            elif self.kind!='memory': raise ValueError(self.kind)
            accept=min(float(np.abs(feature-r).mean()) for r in refs)>self.param
        if accept:
            self.memory[step]=feature
            if len(self.memory)>self.capacity: self.memory.popitem(last=False)
        return accept,list(self.memory)

def configurations():
    return [('all',0)]+[('uniform',k) for k in (2,3,4)]+[(k,t) for k in ('last','memory') for t in THRESHOLDS]

def textset(rows):
    return {re.sub(r'\W+','',r['text'].lower()) for r in rows
            if r['confidence']>=.7 and len(re.sub(r'\W+','',r['text']))>=4
            and .05 < r['bbox'][1] < .95}

def evidence_events(frame_texts):
    occurrences={}
    for i, texts in enumerate(frame_texts):
        for text in texts: occurrences.setdefault(text,[]).append(i)
    return [{'text':t,'support':ids,'query_step':max(ids)+1}
            for t,ids in sorted(occurrences.items()) if len(ids)<=2 and max(ids)+1<len(frame_texts)]

def replay(features,kind,param):
    gate=Gate(kind,param)
    selected,states=[],[]
    for i,f in enumerate(features):
        accept,state=gate.push(i,f)
        if accept: selected.append(i)
        states.append(state)
    return selected,states

def score(events,states):
    missed=[e for e in events if not set(e['support']).intersection(states[e['query_step']])]
    return {'events':len(events),'hits':len(events)-len(missed),'missed':missed}
