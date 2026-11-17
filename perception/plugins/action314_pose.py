"""Darvin #314 pose tool: full MP4 inference and ROS2 camera compatibility.

The shipped development score applies to full MP4 inference. Streaming uses
only its observed prefix and has a separate, explicitly limited scope.
"""
from __future__ import annotations
import base64,collections,json,logging,os,queue,re,tempfile,threading,time,urllib.parse,urllib.request
from pathlib import Path
import cv2,numpy as np
from plugins.action314.mp4_runtime import MP4Recognizer
from plugins.action314.lite_pose import prepare_frame,decode_pose
from plugins.action314.model_store import ensure_models

log=logging.getLogger(__name__)
_VIDEO_SUFFIXES={'.mp4','.mov','.avi','.mkv','.webm','.m4v'}
TOOLS=[{'name':'pose','description':'Recognize one of 126 actions in a full MP4 video. Returns activity, activity_zh and activity_score. Supports ROS2 camera streams; full-video score does not apply to streaming prefixes.',
 'type':'processor','multiInstance':True,'inputSchema':{'type':'object','properties':{
  'action':{'type':'string','enum':['start','stop','info','config','list_actions','recognize_by_video','recognize_video','recognize_by_photo','recognize_by_url','reset']},
  'video_path':{'type':'string','format':'file','accept':'video/*','uploadTo':'mcp'},
  'image_path':{'type':'string','format':'file','accept':'image/*,video/*','uploadTo':'mcp'},
  'file_path':{'type':'string'},'url':{'type':'string'},'input_topic':{'type':'string'},'instance_id':{'type':'string'}},'required':['action'],
 'x-action-params':{'recognize_by_video':{'params':['video_path'],'description':'Full MP4 action recognition'},'recognize_by_photo':{'params':['image_path'],'description':'Image or uploaded MP4 compatibility entry'},'recognize_by_url':{'params':['url'],'description':'Download image or MP4, bounded to 64MiB'},'start':{'params':['input_topic']},'stop':{'params':[]},'info':{'params':[]},'list_actions':{'params':[]}}}}]

class PosePerceptionPlugin:
 PREFIX='pose'
 def __init__(self,plugin_cfg,namespace,executor):
  self.cfg=dict(plugin_cfg);self.namespace=namespace;self.executor=executor;self._runtime=None;self._lock=threading.RLock();self._nodes={};self._last=None;self._pub_node=None;self._pub=None
 def get_tools(self):return TOOLS
 def _model(self):
  with self._lock:
   if self._runtime is None:self._runtime=MP4Recognizer(ensure_models())
   return self._runtime
 def _from_poses(self,poses):
  model=self._model();values=np.asarray(poses,dtype=np.float32);values=values[np.linspace(0,len(values)-1,100).round().astype(int)][None];prob=np.zeros(126,dtype=np.float64)
  for session,spec in zip(model.bodies,model.config['classifiers']):
   logits=session.run(None,{'pose':values})[0][0].astype(np.float64)/spec['temperature'];logits-=logits.max();p=np.exp(logits);p/=p.sum();prob+=spec['weight']*p
  index=int(prob.argmax());label=model.config['labels'][index]
  return {'timestamp':time.time(),'count':1,'persons':[{'id':1,'position':[0.,0.],'activity':label['activity'],'activity_zh':label['activity_zh'],'activity_score':float(prob[index])}],'latency_ms':0.}
 def _pose(self,frame):
  x,g=prepare_frame(frame);return decode_pose(self._model().front.run(None,{'images':x})[0],g)[0]
 def _recognize_path(self,path):
  path=Path(path)
  if not path.is_file():raise ValueError('Input path is not a file')
  with self._lock:
   if path.suffix.lower() in _VIDEO_SUFFIXES:
    result,_=self._model().classify(path)
   else:
    frame=cv2.imdecode(np.fromfile(str(path),dtype=np.uint8),cv2.IMREAD_COLOR)
    if frame is None:result,_=self._model().classify(path)
    else:
     start=time.perf_counter();result=self._from_poses([self._pose(frame)]);result['latency_ms']=(time.perf_counter()-start)*1000
   self._last=result;self._publish(result);return result
 def _publish(self,result):
  if self.executor is None:return
  import rclpy
  from std_msgs.msg import String
  if self._pub_node is None:
   self._pub_node=rclpy.create_node('action314_oneshot');self._pub=self._pub_node.create_publisher(String,'/perception/pose',10);self.executor.add_node(self._pub_node)
  self._pub.publish(String(data=json.dumps(result)))
 def _recognize(self,args):
  path=next((args.get(n) for n in ['video_path','image_path','file_path','path','video'] if args.get(n)),None)
  if path:return self._recognize_path(path)
  url=args.get('url')
  if not url:raise ValueError('video_path, image_path, file_path or url is required')
  parsed=urllib.parse.urlparse(url)
  if parsed.scheme not in {'http','https'}:raise ValueError('Only http(s) URL inputs are supported')
  upload_root=Path(os.environ.get('ACTION314_TEMP_DIR',str(ensure_models()/'.uploads')));upload_root.mkdir(parents=True,exist_ok=True)
  suffix=Path(parsed.path).suffix.lower() or '.mp4'
  with tempfile.NamedTemporaryFile(dir=upload_root,suffix=suffix) as stream:
   with urllib.request.urlopen(url,timeout=30) as response:
    count=0
    while True:
     block=response.read(1024*1024)
     if not block:break
     count+=len(block)
     if count>64*1024*1024:raise ValueError('Input URL exceeds 64MiB')
     stream.write(block)
   stream.flush();return self._recognize_path(stream.name)
 def dispatch(self,name,args):
  action=args.get('action',name)
  if action=='pose':action='recognize_by_video' if any(args.get(n) for n in ['video_path','image_path','file_path','path','url']) else 'info'
  try:
   if action in {'recognize_by_video','recognize_video','recognize_by_photo','recognize_by_url','recognize'}:return self._recognize(args)
   if action=='list_actions':
    labels=json.loads((Path(__file__).parent/'action314/runtime.json').read_text())['labels'];return {'ok':True,'activities':[{'name':r['activity'],'name_zh':r['activity_zh']} for r in labels],'count':126}
   if action=='info':return {'state':'running' if self._nodes else 'ready','backend':'action314_onnx_cpu','model_loaded':self._runtime is not None,'supported_actions':126,'output_topic':'/perception/pose','instances':list(self._nodes),'stream_errors':sum(n.error_count for n in self._nodes.values()),'last_result':self._last,'full_video_only_score':True,'model_bytes':23208501,'jp5_verified':False}
   if action=='config':return {'status':'configured','backend':'action314_onnx_cpu','note':'Fixed validated model, preprocessing and sampling are retained.'}
   if action=='start':
    self._model();topic=args.get('input_topic','');key=args.get('instance_id') or topic or '_default'
    if not topic:return {'state':'running','mode':'on_demand','output_topic':'/perception/pose'}
    if key not in self._nodes:self._nodes[key]=_Camera(self,topic,key)
    return {'state':'running','input_topic':topic,'output_topic':self._nodes[key].output_topic,'scope':'observed-prefix stream, not full-video validation'}
   if action=='reset':
    for node in self._nodes.values():node.reset()
    self._last=None;return {'ok':True}
   if action=='stop':
    key=args.get('instance_id');keys=[key] if key in self._nodes else list(self._nodes) if not key else []
    for key in keys:self._nodes.pop(key).close()
    return {'state':'idle'}
   return None
  except Exception as error:
   log.exception('Action314 request failed')
   return {'timestamp':time.time(),'count':0,'persons':[],'ok':False,'error':type(error).__name__+': '+str(error),'latency_ms':0.}
 def shutdown(self):
  self.dispatch('pose',{'action':'stop'})
  if self._pub_node is not None:self.executor.remove_node(self._pub_node);self._pub_node.destroy_node();self._pub_node=None;self._pub=None

class _Camera:
 def __init__(self,owner,topic,key):
  import rclpy
  from sensor_msgs.msg import CompressedImage,Image
  from std_msgs.msg import String
  from rclpy.qos import qos_profile_sensor_data
  self.owner=owner;self.output_topic=topic.rstrip('/')+'/poses';self._string=String;self.queue=queue.Queue(maxsize=4);self.stop=threading.Event();self.history=collections.deque(maxlen=100);self._last_stamp=None;self._history_lock=threading.Lock();self.error_count=0;self.generation=0
  self.node=rclpy.create_node('action314_'+re.sub('[^A-Za-z0-9_]','_',key)[-60:]);self.publisher=self.node.create_publisher(String,self.output_topic,10);self.sub=self.node.create_subscription(CompressedImage,topic,self._compressed,qos_profile_sensor_data);owner.executor.add_node(self.node);self.thread=threading.Thread(target=self._worker,daemon=True,name='action314-stream');self.thread.start()
 def _compressed(self,msg):
  stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
  if stamp<=0:stamp=time.monotonic()
  with self._history_lock:
   if self._last_stamp is not None and (stamp<self._last_stamp or stamp-self._last_stamp>3):self.history.clear();self._last_stamp=None;self.generation+=1
   if self._last_stamp is not None and stamp-self._last_stamp<.125-1e-6:return
   self._last_stamp=stamp;generation=self.generation
  frame=cv2.imdecode(np.frombuffer(msg.data,np.uint8),cv2.IMREAD_COLOR)
  if frame is None:self.error_count+=1;return
  try:self.queue.put_nowait((generation,frame))
  except queue.Full:
   try:self.queue.get_nowait()
   except queue.Empty:pass
   self.queue.put_nowait((generation,frame))
 def _worker(self):
  while not self.stop.is_set():
   try:generation,frame=self.queue.get(timeout=.2)
   except queue.Empty:continue
   try:
    started=time.perf_counter()
    with self.owner._lock:
     if self.stop.is_set():break
     pose=self.owner._pose(frame)
     with self._history_lock:
      if generation!=self.generation:continue
      self.history.append(pose);poses=list(self.history)
     result=self.owner._from_poses(poses);result['latency_ms']=(time.perf_counter()-started)*1000;self.owner._last=result
    if not self.stop.is_set():self.publisher.publish(self._string(data=json.dumps(result)))
   except Exception:log.exception('Action314 stream frame failed');self.error_count+=1
 def reset(self):
  with self._history_lock:self.history.clear();self._last_stamp=None;self.generation+=1
 def close(self):
  self.stop.set();self.thread.join(timeout=10);self.owner.executor.remove_node(self.node);self.node.destroy_node()
