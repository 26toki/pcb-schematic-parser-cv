"""OCR plus three observed descending bars; proposals only, not deployed."""
import itertools,math
import numpy as np
import cv2
def proposals(image,items):
    a=np.asarray(image.convert('RGB')).astype(np.int16);ink=((a[:,:,0]-a[:,:,1]>35)&(a[:,:,0]-a[:,:,2]>35)&(a[:,:,0]>70)).astype('uint8');found=[]
    eligible=[t for t in items if t['text'] in ('GND','AGND','DGND','PGND') and t['confidence']>=.95]
    for text in eligible[:64]:
        if text['text'] not in ('GND','AGND','DGND','PGND') or text['confidence']<.95:continue
        x,y,u,v=text['bbox_xyxy'];cx,cy=(x+u)/2,(y+v)/2;left=max(0,int(cx-65));top=max(0,int(cy-65));chip=ink[top:min(ink.shape[0],int(cy+66)),left:min(ink.shape[1],int(cx+66))]
        options=[]
        for axis in (0,1):
            opened=cv2.morphologyEx(chip,cv2.MORPH_OPEN,np.ones((1,7) if axis==0 else (7,1),np.uint8));n,labels,stats,centers=cv2.connectedComponentsWithStats(opened,8);bars=[]
            for i in range(1,n):
                px,py,w,h,area=map(int,stats[i]);length,thickness=(w,h) if axis==0 else (h,w)
                if 6<=length<=65 and thickness<=4:bars.append(dict(center=[float(centers[i][0])+left,float(centers[i][1])+top],length=length,bbox=[px+left,py+top,px+w-1+left,py+h-1+top]))
            for triple in itertools.combinations(bars,3):
                ordered=sorted(triple,key=lambda b:-b['length']);long,mid,short=ordered;across=axis;along=1-axis
                if max(b['center'][across] for b in ordered)-min(b['center'][across] for b in ordered)>2:continue
                if not long['length']>mid['length']>short['length'] or short['length']>.75*long['length']:continue
                ds=[mid['center'][along]-long['center'][along],short['center'][along]-mid['center'][along]]
                if ds[0]*ds[1]<=0 or not all(3<=abs(d)<=14 for d in ds) or max(map(abs,ds))/min(map(abs,ds))>1.5:continue
                bx=[min(b['bbox'][0] for b in ordered),min(b['bbox'][1] for b in ordered),max(b['bbox'][2] for b in ordered),max(b['bbox'][3] for b in ordered)]
                if math.dist([cx,cy],long['center'])>60:continue
                options.append(dict(type='gnd',name=text['text'],bbox=bx,bars=ordered,stem_axis='vertical' if axis==0 else 'horizontal',ocr_id=text['id'],evidence='high_confidence_GND_text_and_three_aligned_descending_red_bars'))
        options.sort(key=lambda r:math.dist([cx,cy],r['bars'][0]['center']))
        if options and (len(options)==1 or options[0]['bbox']==options[1]['bbox']):found.append(options[0])
    return found
