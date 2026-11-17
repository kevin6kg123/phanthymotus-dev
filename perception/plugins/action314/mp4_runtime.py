"""CPU ONNX MP4 inference; all pose and classification work is executed locally."""
import hashlib,json,logging,math,os,time
from pathlib import Path
import cv2,numpy as np,onnxruntime as ort
try:
 from .lite_pose import prepare_frame,decode_pose
except ImportError:
 from lite_pose import prepare_frame,decode_pose

class MP4Recognizer:
 def __init__(self,root,config='runtime.json'):
  self.root=Path(root).resolve();self.config=json.loads((self.root/config).read_text());c=self.config
  cv2.setNumThreads(1)
  for name in [c['pose_model'],*[x['file'] for x in c['classifiers']]]:
   path=(self.root/name).resolve()
   if not path.is_relative_to(self.root):raise ValueError('Model path escaped bundle')
   if hashlib.sha256(path.read_bytes()).hexdigest()!=c['model_sha256'][name]:raise ValueError('Model hash mismatch: '+name)
  want=os.environ.get('ACTION314_DEVICE','cuda').lower();self.device='cpu';self.device_error=None
  if want=='cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
   try:self._build(['cuda']);self.device='cuda'
   except Exception as error:self.device_error=type(error).__name__+': '+str(error)[:300];logging.getLogger(__name__).warning('[action314] CUDA provider unavailable, using CPU: %s',self.device_error)
  if self.device!='cuda':self._build(['cpu'])
  labels=[x['activity'] for x in c['labels']];self.none_index=labels.index('none') if 'none' in labels and os.environ.get('ACTION314_ALLOW_NONE')!='1' else None
 def _session(self,name,kind):
  options=ort.SessionOptions();options.inter_op_num_threads=1
  if kind=='cuda':
   options.intra_op_num_threads=1
   providers=[('CUDAExecutionProvider',{'device_id':0,'gpu_mem_limit':int(os.environ.get('ACTION314_GPU_MEM_MB','192'))*1024*1024,'arena_extend_strategy':'kSameAsRequested','cudnn_conv_algo_search':'HEURISTIC','do_copy_in_default_stream':True}),'CPUExecutionProvider']
  else:
   options.intra_op_num_threads=int(os.environ.get('ACTION314_CPU_THREADS','1'));providers=['CPUExecutionProvider']
  session=ort.InferenceSession(str(self.root/name),sess_options=options,providers=providers)
  if kind=='cuda' and session.get_providers()[0]!='CUDAExecutionProvider':raise RuntimeError('CUDA provider not active: '+str(session.get_providers()))
  return session
 def _build(self,kinds):
  c=self.config;kind=kinds[0]
  front=self._session(c['pose_model'],kind);bodies=[self._session(x['file'],kind) for x in c['classifiers']]
  import numpy as _np
  front.run(None,{'images':_np.zeros((1,3,416,416),_np.float32)})
  for b in bodies:b.run(None,{'pose':_np.zeros((1,c.get('pose_frames',100),17,3),_np.float32)})
  self.front,self.bodies=front,bodies
  assert len(c['labels'])==126 and math.isclose(sum(x['weight'] for x in c['classifiers']),1.)

 def classify(self,path):
  start=time.perf_counter();c=self.config;option=getattr(cv2,'CAP_PROP_N_THREADS',None)
  cap=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[option,1]) if option else cv2.VideoCapture(str(path))
  if not cap.isOpened():raise ValueError('Cannot open video')
  total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
  if total<1 or not math.isfinite(fps) or fps<=0:cap.release();raise ValueError('Invalid video metadata')
  count=min(total,c.get('max_pose_frames',100),max(2,math.ceil(total/fps*c.get('pose_hz',8))));wanted=set(np.linspace(0,total-1,count).round().astype(int).tolist());poses=[];timestamps=[];index=0;missing=0
  try:
   while True:
    ok,frame=cap.read()
    if not ok:break
    if index in wanted:
     x,g=prepare_frame(frame);raw=self.front.run(None,{'images':x})[0];pose,score=decode_pose(raw,g);timestamp=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
     if not math.isfinite(timestamp) or timestamp<0 or (timestamps and timestamp<=timestamps[-1]):raise ValueError('Invalid decoder timestamps')
     poses.append(pose);timestamps.append(timestamp);missing+=int(score<.25)
    index+=1
  finally:cap.release()
  if index!=total or len(poses)!=len(wanted):raise ValueError('Incomplete video decode')
  pose=np.stack(poses)[np.linspace(0,len(poses)-1,c.get('pose_frames',100)).round().astype(int)][None].astype(np.float32)
  probs=np.zeros((1,126),np.float64);all_logits=[]
  for session,spec in zip(self.bodies,c['classifiers']):
   logits=session.run(None,{'pose':pose})[0]
   if logits.shape!=(1,126) or not np.isfinite(logits).all():raise ValueError('Invalid classifier logits')
   scaled=logits.astype(np.float64)/spec.get('temperature',1.);scaled-=scaled.max(1,keepdims=True);p=np.exp(scaled);p/=p.sum(1,keepdims=True);probs+=spec['weight']*p;all_logits.append(logits[0].tolist())
  if self.none_index is not None:probs[0,self.none_index]=0.
  prediction=int(probs.argmax(1)[0]);label=c['labels'][prediction];elapsed=(time.perf_counter()-start)*1000
  response={'timestamp':time.time(),'count':1,'persons':[{'id':1,'position':[0.,0.],'activity':label['activity'],'activity_zh':label['activity_zh'],'activity_score':float(probs[0,prediction])}],'latency_ms':elapsed}
  return response,{'source_frames':index,'processed_frames':len(poses),'missing_pose_frames':missing,'prediction':prediction,'latency_ms':elapsed,'classifier_logits':all_logits,'probabilities':probs[0].tolist()}

def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('video');p.add_argument('--root',default=str(Path(__file__).resolve().parent));a=p.parse_args();recognizer=MP4Recognizer(a.root)
 try:result,_=recognizer.classify(a.video)
 except ValueError as e:result={'timestamp':time.time(),'count':0,'persons':[],'error':str(e)}
 print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
