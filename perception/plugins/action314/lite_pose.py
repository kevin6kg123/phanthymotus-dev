"""Shared single-actor letterbox and raw COCO17 decoding; no detector dependency."""
import numpy as np


def prepare_frame(frame,size=416):
    import cv2
    h,w=frame.shape[:2];gain=min(size/w,size/h);nw,nh=round(w*gain),round(h*gain);left=(size-nw)//2;top=(size-nh)//2
    resized=cv2.resize(frame,(nw,nh));canvas=np.full((size,size,3),114,dtype=np.uint8);canvas[top:top+nh,left:left+nw]=resized
    tensor=np.ascontiguousarray(canvas[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255
    return tensor,{'width':w,'height':h,'gain':gain,'left':left,'top':top}


def decode_pose(output,geometry,threshold=.25):
    prediction=np.asarray(output)[0]
    if prediction.shape[0]!=56 or not np.isfinite(prediction).all():raise ValueError('Expected finite raw single-class COCO17 pose output')
    index=int(prediction[4].argmax());score=float(prediction[4,index]);pose=np.zeros((17,3),dtype=np.float32)
    if score<threshold:return pose,score
    points=prediction[5:,index].reshape(17,3);pose[:,0]=((points[:,0]-geometry['left'])/geometry['gain'])/(geometry['width']/2)-1;pose[:,1]=((points[:,1]-geometry['top'])/geometry['gain'])/(geometry['height']/2)-1
    pose[:,:2]=np.clip(pose[:,:2],-1,1);pose[:,2]=np.clip(points[:,2]*score,0,1)
    return pose,score
