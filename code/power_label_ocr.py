"""Local rotated OCR with coordinates mapped back to the original PNG; no GT/API."""
from PIL import Image,ImageDraw


def inverse_rotation(x,y,angle,width,height):
    if angle==0:return x,y
    if angle==90:return width-y,x
    if angle==180:return width-x,height-y
    if angle==270:return y,height-x
    raise ValueError('Right angles required')


def local_labels(engine,image,detection,ics,scale=3):
    b=detection.get('raw_bbox_xyxy',detection['body_bbox_xyxy'])
    region=(max(0,int(b[0])-48),max(0,int(b[1])-48),min(image.width,int(b[2])+49),min(image.height,int(b[3])+49))
    left,top,right,bottom=region;patch=image.crop(region).convert('RGB')
    draw=ImageDraw.Draw(patch)
    for ic in ics:
        x,y,u,v=ic['body_bbox_xyxy']
        if x<right and u>left and y<bottom and v>top:
            draw.rectangle((max(0,x-left),max(0,y-top),min(patch.width,u-left),min(patch.height,v-top)),fill='white')
    records=[]
    for angle in (0,90,270,180):
        rotated=patch.rotate(angle,expand=True)
        resized=rotated.resize((rotated.width*scale,rotated.height*scale),Image.Resampling.LANCZOS)
        import numpy as np
        result,_=engine(np.asarray(resized))
        for quad,text,score in result or []:
            points=[inverse_rotation(float(x)/scale,float(y)/scale,angle,patch.width,patch.height) for x,y in quad]
            points=[[x+left,y+top] for x,y in points];xs,ys=zip(*points)
            compact=''.join(text.split())
            records.append(dict(id=f"local_{detection['id']}_{angle}_{len(records)}",text=compact,raw_text=text,
                                confidence=float(score),bbox_xyxy=[min(xs),min(ys),max(xs),max(ys)],
                                source='local_rotated_ocr',rotation=angle,crop_xyxy=list(region)))
    return records
