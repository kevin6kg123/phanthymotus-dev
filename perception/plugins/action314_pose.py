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
TOOLS[0].update(topic_in=[{'format':'image/jpeg','desc':'camera image input'}],topic_out=[{'format':'data/json','desc':'126-class action labels and confidence'}])

class PosePerceptionPlugin:
 PREFIX='pose'
 def __init__(self,plugin_cfg,namespace,executor):
  self.cfg=dict(plugin_cfg);self.namespace=namespace;self.executor=executor;self._runtime=None;self._lock=threading.RLock();self._nodes={};self._last=None;self._pub_node=None;self._pub=None;self._on_demand_started=False;self._load_error=None
  threading.Thread(target=self._preload,daemon=True,name='action314-preload').start()
 def _preload(self):
  try:
   started=time.perf_counter();self._model();log.info('[action314] models loaded in %.2fs',time.perf_counter()-started)
  except Exception as error:
   self._load_error=type(error).__name__+': '+str(error);log.exception('[action314] model preload failed')
 def get_tools(self):return TOOLS
 def _model(self):
  with self._lock:
   if self._runtime is None:self._runtime=MP4Recognizer(ensure_models())
   return self._runtime
 def _from_poses(self,poses):
  model=self._model();values=np.asarray(poses,dtype=np.float32);values=values[np.linspace(0,len(values)-1,100).round().astype(int)][None];prob=np.zeros(126,dtype=np.float64)
  for session,spec in zip(model.bodies,model.config['classifiers']):
   logits=session.run(None,{'pose':values})[0][0].astype(np.float64)/spec['temperature'];logits-=logits.max();p=np.exp(logits);p/=p.sum();prob+=spec['weight']*p
  if model.none_index is not None:prob[model.none_index]=0.
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
  upload_root=Path(os.environ.get('ACTION314_TEMP_DIR',tempfile.gettempdir()));upload_root.mkdir(parents=True,exist_ok=True)
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
   if action=='info':return self._info(args)
   if action=='info_legacy':return {'state':'running' if self._nodes or self._on_demand_started else 'ready','backend':'action314_onnx_cpu','model_loaded':self._runtime is not None,'load_error':self._load_error,'supported_actions':126,'output_topic':'/perception/pose','instances':list(self._nodes),'stream_errors':sum(n.error_count for n in self._nodes.values()),'last_result':self._last,'full_video_only_score':True,'model_bytes':23208501,'jp5_verified':False}
   if action=='config':
    iid=args.get('instance_id') or '';cfg={k:v for k,v in args.items() if k not in ('action','instance_id') and v is not None and v!=''}
    return {'status':'configured','instance_id':iid,'config':cfg} if iid else {'status':'configured','config':cfg}
   if action=='start':
    topic=args.get('input_topic','');key=args.get('instance_id') or topic or '_default'
    if not topic:self._on_demand_started=True;return {'state':'running','mode':'on_demand','output_topic':'/perception/pose'}
    if key not in self._nodes:self._nodes[key]=_Camera(self,topic,key)
    return self._nodes[key].status()
   if action=='reset':
    for node in self._nodes.values():node.reset()
    self._last=None;return {'ok':True}
   if action=='stop':
    instance_key=args.get('instance_id');keys=[instance_key] if instance_key in self._nodes else list(self._nodes) if not instance_key else []
    last=None
    for key in keys:
     node=self._nodes.pop(key);node.close();last=node.status(running=False)
    if not instance_key:self._on_demand_started=False
    return last if (instance_key and last) else ({'state':'idle','stopped_instances':keys} if keys else {'state':'idle'})
   return None
  except Exception as error:
   log.exception('Action314 request failed')
   return {'timestamp':time.time(),'count':0,'persons':[],'ok':False,'error':type(error).__name__+': '+str(error),'latency_ms':0.}
 def _info(self,args):
  nodes=dict(self._nodes)
  base={'name':'PosePerception','manufacture':'Embodied','model':'action314_onnx_cpu'}
  if self._runtime is None and not self._load_error:return {**base,'state':'loading','desc':'Loading pose engine...'}
  if self._load_error:return {**base,'state':'error','desc':'Engine load failed: '+str(self._load_error)}
  instances={k:n.info() for k,n in nodes.items()}
  iid=args.get('instance_id') or '';topic=args.get('input_topic') or ''
  if iid in nodes:topic=nodes[iid].topic
  elif not topic and nodes:topic=next(iter(nodes.values())).topic
  out=topic.rstrip('/')+'/poses' if topic else '/perception/pose'
  info={**base,'state':'running' if instances else 'idle','keypoints':17,
   'keypoint_names':['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle'],
   'action_backend':'action314','action_backend_effective':'action314','postures':[],'instances':instances,
   'topic_in':[{'topic':topic,'format':'image/jpeg'}] if topic else [],
   'topic_out':[{'topic':out,'format':'data/json'},{'topic':out+'/skeleton','format':'sensor/pose2d'}] if (topic or nodes) else [],
   'desc':'126-class action recognition (CPU ONNX pose + ST-GCN++)','hands':False,
   'backend':'action314_onnx_cpu','model_loaded':self._runtime is not None,'supported_actions':126,'device':getattr(self._runtime,'device',None),'device_error':getattr(self._runtime,'device_error',None),'last_result':self._last}
  preds={k:n.last_prediction for k,n in nodes.items() if n.last_prediction}
  if preds:info['last_prediction']=preds
  return info
 def shutdown(self):
  self.dispatch('pose',{'action':'stop'})
  if self._pub_node is not None:self.executor.remove_node(self._pub_node);self._pub_node.destroy_node();self._pub_node=None;self._pub=None

class _Camera:
 def __init__(self,owner,topic,key):
  import rclpy
  from sensor_msgs.msg import CompressedImage,Image
  from std_msgs.msg import String
  from rclpy.qos import qos_profile_sensor_data
  self.owner=owner;self.topic=topic;self._result=None;self.detect_count=0;self.person_count=0;self.last_actions={};self.last_prediction={};self.fps=30;self.output_topic=topic.rstrip('/')+'/poses';self._string=String;self.queue=queue.Queue(maxsize=4);self.stop=threading.Event();self.history=collections.deque(maxlen=100);self._last_stamp=None;self._history_lock=threading.Lock();self.error_count=0;self.generation=0
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
     self._since=getattr(self,'_since',0)+1
     if self._result is None or self.queue.empty() or self._since>=3:
      self._since=0;result=self.owner._from_poses(poses);self._result=result
     else:
      result=dict(self._result);result['timestamp']=time.time()
     result['latency_ms']=int((time.perf_counter()-started)*1000);self.owner._last=result
     person=result['persons'][0] if result.get('persons') else None
     self.last_actions={person['id']:person['activity']} if person else {}
     self.last_prediction={'activity':person['activity'],'activity_zh':person['activity_zh'],'score':person['activity_score']} if person else {}
     self.detect_count+=1;self.person_count=len(result.get('persons') or [])
    if not self.stop.is_set():self.publisher.publish(self._string(data=json.dumps(result)))
   except Exception:log.exception('Action314 stream frame failed');self.error_count+=1
 def status(self,running=True):
  return {'state':'running' if running else 'idle','input':self.topic,'output':self.output_topic,'skeleton_output':self.output_topic+'/skeleton','overlay_output':None,'gesture_output':None,'mode':'stream','hands':False}
 def info(self):
  return {'input':self.topic,'output':self.output_topic,'skeleton_output':self.output_topic+'/skeleton','overlay_output':None,'confidence':0.25,'fps':self.fps,'kpt_confidence':0.0,'max_persons':1,'publish_keypoints':'off','publish_bbox':False,'publish_overlay':False,'detect_count':self.detect_count,'persons_last_frame':self.person_count,'last_actions':dict(self.last_actions),'gesture_output':None,'gesture_whitelist':[],'gesture_event_count':0,'recent_gestures':[],'hands':False,'stream_errors':self.error_count}
 def reset(self):
  with self._history_lock:self.history.clear();self._last_stamp=None;self.generation+=1
 def close(self):
  self.stop.set();self.thread.join(timeout=10);self.owner.executor.remove_node(self.node);self.node.destroy_node()
