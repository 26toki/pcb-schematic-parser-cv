"""Observe a continuous narrow axis through compact, pixel-supported junction dots."""
import numpy as np

def reflected_dot_iou(mask,point,direction,core):
    """Reject asymmetric symbol silhouettes; allow one-pixel raster centering."""
    x,y=map(round,point);best=0.
    for delta in (-1,0,1):
        xx,yy=(x+delta,y) if direction=='horizontal' else (x,y+delta)
        if min(xx,yy)<12 or xx+12>=mask.shape[1] or yy+12>=mask.shape[0]:continue
        p=mask[yy-12:yy+13,xx-12:xx+13].copy()
        if direction=='vertical':p=p.T
        half=core//2+1;p[12-half:13+half]=False
        reflected=p[:,::-1];union=int((p|reflected).sum())
        if union:best=max(best,float((p&reflected).sum())/union)
    return best

def axial_band(component,origin,direction,original,dot_evidence):
    a=np.asarray(component,dtype=bool)
    horizontal=direction=='horizontal';view=a if horizontal else a.T
    thick,length=view.shape
    if not 10<thick<=24 or length<max(24,3*thick):return None
    core=np.flatnonzero(view.mean(axis=1)>=.95)
    if not 1<=len(core)<=6 or np.any(np.diff(core)!=1):return None
    supported=np.flatnonzero(view[core].mean(axis=0)>=.66)
    if not len(supported) or np.any(np.diff(supported)!=1):return None
    start,end=int(supported[0]),int(supported[-1])
    # Trim only short rounded caps; never bridge an interior whitespace gap.
    if start>len(core) or length-1-end>len(core):return None
    mid=float(core.mean());lo,hi=int(core[0]),int(core[-1])
    upper=view[:max(0,lo-2)];lower=view[min(thick,hi+3):]
    up,down=int(upper.sum()),int(lower.sum())
    if not up or not down or not .5<=up/down<=2:return None
    if abs(lo-(thick-1-hi))>2:return None
    extra=view.copy();extra[max(0,lo-2):min(thick,hi+3)]=False
    columns=np.flatnonzero(extra.any(axis=0))
    if not len(columns):return None
    groups=np.split(columns,np.flatnonzero(np.diff(columns)>1)+1)
    if not 1<=len(groups)<=3:return None
    dots=[];reflection_scores=[];x,y=origin
    for g in groups:
        if int(g[-1]-g[0]+1)>max(18,4*len(core)+8):return None
        along=float((g[0]+g[-1])/2)
        point=[x+along,y+mid] if horizontal else [x+mid,y+along]
        if not dot_evidence(original,*point):return None
        reflection=reflected_dot_iou(original,point,direction,len(core))
        if reflection<.80:return None
        reflection_scores.append(reflection)
        dots.append(point)
    return dict(axis=(y if horizontal else x)+mid,start_offset=start,end_offset=end,source='continuous_narrow_axis_through_observed_compact_dots',core_thickness_px=len(core),discarded_component_thickness_px=thick,observed_dots_xy=dots,along_axis_reflection_iou=reflection_scores)
