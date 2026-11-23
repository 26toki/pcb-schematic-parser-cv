"""Conservative chroma-halo fallback with dark red core and center green wire."""
import numpy as np
def robust_tip(rgb,pin):
    if rgb is None or pin.get('body_contact_xy') is None:return None
    x,y=map(round,pin['body_contact_xy']);vertical=pin['side'] in ('top','bottom');sign=-1 if pin['side'] in ('left','top') else 1;labels=[];center_red=[];center_green=[]
    for gap in range(1,81):
        xx,yy=(x,y+sign*gap) if vertical else (x+sign*gap,y)
        if not (1<=xx<rgb.shape[1]-1 and 1<=yy<rgb.shape[0]-1):break
        strip=(rgb[yy,xx-1:xx+2] if vertical else rgb[yy-1:yy+2,xx]).astype(float);lum=strip@np.array([.299,.587,.114]);core=strip[int(np.argmin(lum))];r,g,b=core
        red=r-g>=12 and r-b>=12 and (r-g)/(r+g+1)>=.10 and r>40
        green=g-r>=12 and g-b>=12 and (g-r)/(g+r+1)>=.07 and g>40
        cr,cg,cb=rgb[yy,xx].astype(float);center_red.append(cr-cg>=12 and cr-cb>=12 and (cr-cg)/(cr+cg+1)>=.10 and cr>40);center_green.append(cg-cr>=12 and cg-cb>=12 and (cg-cr)/(cg+cr+1)>=.07 and cg>40 and .299*cr+.587*cg+.114*cb<170)
        labels.append('r' if red else 'g' if green else '?')
    if len(labels)<12 or sum(center_red[:6])<4:return None
    for index in range(8,len(labels)-5):
        if labels[index:index+6]!=['g']*6 or not all(center_green[index:index+6]):continue
        end=index-1
        while end>=0 and labels[end]=='?' and index-end<=3:end-=1
        if end<5 or labels[end-5:end+1]!=['r']*6:return None
        if 'g' in labels[:end]:return None
        gap=index+1
        return dict(status='observed_red_pin_green_wire_boundary',position_image_xy=[x if vertical else x+sign*gap,y+sign*gap if vertical else y],gap_px=gap,source='dark_red_core_then_six_center_green_pixels',evidence=dict(red_core_run_min=6,center_green_run_min=6,center_red_near_body_min=4,max_antialias_gap=3))
    return None
