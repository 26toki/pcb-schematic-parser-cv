"""Choose a reference correction only from aligned, exact two-view evidence."""
import re,math
def choose(old,local,second,used):
    if not old.get('text') or old.get('confidence',1)>=.85:return None
    b=old['bbox_xyxy'];cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2;names={}
    def valid(t):
        q=t['bbox_xyxy'];qx,qy=(q[0]+q[2])/2,(q[1]+q[3])/2
        return t.get('confidence',0)>=.97 and re.fullmatch(r'[A-Za-z]{1,6}\d+[A-Za-z0-9]*',t['text']) and math.dist([cx,cy],[qx,qy])<=max(3,.5*min(b[3]-b[1],q[3]-q[1]))
    for a in local:
        if not valid(a) or a['text']==old['text'] or a['text'] in used:continue
        for c in second:
            if valid(c) and c['text']==a['text']:names[a['text']]=dict(before=old['text'],after=a['text'],evidence=[a,c])
    return next(iter(names.values())) if len(names)==1 else None
