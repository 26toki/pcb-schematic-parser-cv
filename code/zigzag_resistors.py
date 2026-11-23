"""Alternating two-sided angular strokes and external leads, without GT inputs."""
import math
import cv2
import numpy as np

def zigzag_geometries(mask,excluded_boxes):
    from observe_passive_terminals import branch
    ink=mask.astype(np.uint8);straight=np.zeros_like(ink)
    for kernel in [(14,1),(1,14)]:
        straight|=cv2.morphologyEx(ink,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,kernel))
    curved=ink & (1-straight);n,labels,stats,_=cv2.connectedComponentsWithStats(curved,8);result=[]
    for index in range(1,n):
        x,y,w,h,area=map(int,stats[index]);horizontal=w>h;long=max(w,h);short=min(w,h)
        if not (20<=long<=140 and 6<=short<=40 and 1.7<=long/short<=12):continue
        if any(l<=x+w/2<=r and t<=y+h/2<=b for l,t,r,b in excluded_boxes):continue
        patch=labels[y:y+h,x:x+w]==index;view=patch if horizontal else patch.T
        baselines=[]
        left=np.flatnonzero(view[:,:2].any(axis=1));right=np.flatnonzero(view[:,-2:].any(axis=1))
        if len(left) and len(right) and abs(np.median(left)-np.median(right))<=3:
            baselines.append(float((np.median(left)+np.median(right))/2))
        # Keep the existing successful measurement first. If clipped shoulders
        # fail, try an axis independently observed in both external leads.
        original=ink if horizontal else ink.T
        u,v,extent,depth=(x,y,w,h) if horizontal else (y,x,h,w)
        supports=[]
        if u>=8 and u+extent+8<=original.shape[1]:
            for a,b in [(u-8,u-2),(u+extent+2,u+extent+8)]:
                coverage=original[v:v+depth,a:b].mean(axis=1)
                rows=np.flatnonzero(coverage>=.8)
                if not len(rows) or len(rows)>5 or np.any(np.diff(rows)>1):break
                supports.append(float(np.median(rows)))
        if len(supports)==2 and abs(supports[0]-supports[1])<=2:
            observed=float(sum(supports)/2)
            if observed not in baselines:baselines.append(observed)
        for baseline in baselines:
            profile=[]
            for col in range(view.shape[1]):
                rows=np.flatnonzero(view[:,col])
                if not len(rows):break
                profile.append(float(np.mean(rows))-baseline)
            if len(profile)!=view.shape[1]:continue
            p=np.convolve(profile,np.ones(3)/3,mode='same')
            if p.max()<3 or p.min()>-3:continue
            threshold=max(2,.3*(p.max()-p.min())/2);lobes=[]
            for sign in [1,-1]:
                high=p*sign>=threshold;edges=np.diff(np.r_[False,high,False].astype(np.int8))
                for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
                    if b-a>=2:lobes.append(((a+b-1)/2,sign))
            lobes.sort()
            if not 4<=len(lobes)<=10 or any(a[1]==b[1] for a,b in zip(lobes,lobes[1:])):continue
            gaps=np.diff([a[0] for a in lobes])
            if min(gaps)<3 or max(gaps)/min(gaps)>2.8:continue
            active=np.flatnonzero(np.abs(p)>=threshold)
            first=int(active[0]);last=int(active[-1])
            axis='horizontal' if horizontal else 'vertical'
            contacts=[[x+first-1,y+baseline],[x+last+1,y+baseline]] if horizontal else [[x+baseline,y+first-1],[x+baseline,y+last+1]]
            if not branch(mask,axis,contacts[0],-1) or not branch(mask,axis,contacts[1],1):continue
            box=[x+first-1,y-1,x+last+1,y+h] if horizontal else [x-1,y+first-1,x+w,y+last+1]
            result.append(dict(type='r',axis=axis,body_bbox_xyxy=box,terminal_contacts_xy=contacts,
                geometry_source='observed_alternating_zigzag_with_inline_leads',
                zigzag_evidence=dict(alternating_lobes=len(lobes),positive_depth=float(p.max()),negative_depth=float(-p.min()),
                    spacing_ratio=float(max(gaps)/min(gaps)))))
            break
    return result
