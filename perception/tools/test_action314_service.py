"""Run the real perception entry point and exercise its MCP upload/video API.

Use a fresh personal output directory on Jetson. No system package changes.
"""
import argparse,concurrent.futures,functools,hashlib,json,os,re,secrets,socket,subprocess,sys,threading,time,urllib.request
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from pathlib import Path

def main():
 parser=argparse.ArgumentParser(__doc__);parser.add_argument('--samples',required=True);parser.add_argument('--out',required=True);args=parser.parse_args();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False);source=Path(args.samples).resolve();manifest=json.loads((source/'manifest.json').read_text());perception=Path(__file__).resolve().parents[1]
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));mcp_port=sock.getsockname()[1]
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));ws_port=sock.getsockname()[1]
 import yaml
 config=yaml.safe_load((perception/'config.yaml').read_text());config['register_with_agent_core']=False;config['file_intake']['dir']=str(out/'uploads');(out/'test-config.yaml').write_text(yaml.safe_dump(config));test_token=secrets.token_urlsafe(24);env=os.environ.copy();env.update(ACCESS_TOKEN=test_token,CONFIG_PATH=str(out/'test-config.yaml'),MCP_PORT=str(mcp_port),WS_PORT=str(ws_port),ROS_DOMAIN_ID='74',ROS_LOCALHOST_ONLY='1',ROS_LOG_DIR=str(out/'ros-logs'),XDG_CACHE_HOME=str(out/'.cache'),ACTION314_TEMP_DIR=str(out/'url-downloads'));log=(out/'service.log').open('w');server=subprocess.Popen([sys.executable,str(perception/'main.py')],cwd=perception,env=env,stdout=log,stderr=subprocess.STDOUT);url=f'http://127.0.0.1:{mcp_port}/mcp';records=[];report={'scope':'real upstream perception/main.py MCP HTTP and ROS2/WS test with task profile','ports':{'MCP_PORT':mcp_port,'WS_PORT':ws_port},'main_sha256':hashlib.sha256((perception/'main.py').read_bytes()).hexdigest()};monitor=None;monitor_log=None;fileserver=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(SimpleHTTPRequestHandler,directory=str(source)));threading.Thread(target=fileserver.serve_forever,daemon=True).start()
 def rpc(method,params=None):
  request={'jsonrpc':'2.0','id':1,'method':method,'params':params or {}};raw=json.dumps(request).encode();response=json.load(urllib.request.urlopen(urllib.request.Request(url,data=raw,headers={'Content-Type':'application/json'}),timeout=60))
  if response.get('error'):raise RuntimeError(str(response['error']))
  return response['result']
 def call(name,arguments):return json.loads(rpc('tools/call',{'name':name,'arguments':arguments})['content'][0]['text'])
 def validate(result,expected=None):
  assert result['count']==1 and len(result['persons'])==1 and result['latency_ms']>=0;person=result['persons'][0];assert person['id']==1 and len(person['position'])==2 and 0<=person['activity_score']<=1
  labels={r['activity']:r['activity_zh'] for r in manifest['labels']};assert labels[person['activity']]==person['activity_zh']
  if expected is not None:assert person['activity']==manifest['labels'][expected]['activity']
 def upload(path):
  boundary='action314-upload-boundary';data=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\nContent-Type: video/mp4\r\n\r\n').encode()+path.read_bytes()+f'\r\n--{boundary}--\r\n'.encode();request=urllib.request.Request(f'http://127.0.0.1:{mcp_port}/file/upload',data=data,headers={'Content-Type':f'multipart/form-data; boundary={boundary}','X-Access-Token':test_token});j=json.load(urllib.request.urlopen(request,timeout=30));assert j['ok'] and j['bytes']==path.stat().st_size;return j['path']
 try:
  deadline=time.monotonic()+45
  while True:
   if server.poll() is not None:raise RuntimeError('Perception entry point exited before ready')
   try:rpc('initialize');break
   except OSError:
    if time.monotonic()>deadline:raise TimeoutError('MCP server startup timed out')
    time.sleep(.1)
  tools=rpc('tools/list')['tools'];assert any(t['name']=='pose' for t in tools);catalogue=call('pose',{'action':'list_actions'});assert catalogue['count']==126 and len(catalogue['activities'])==126;report['label_catalogue_exact_match']=catalogue['activities']==[{'name':r['activity'],'name_zh':r['activity_zh']} for r in manifest['labels']];assert report['label_catalogue_exact_match']
  from websockets.sync.client import connect
  with connect(f'ws://127.0.0.1:{ws_port}',open_timeout=10) as ws:
   notice=json.loads(ws.recv());assert notice['type']=='asr_error' and 'disabled' in notice['payload']['error']
  report['ws_port_handshake_passed']=True
  if os.path.isfile('/etc/nv_tegra_release'):
   monitor_log=(out/'tegrastats.txt').open('w');monitor=subprocess.Popen(['timeout','180','tegrastats','--interval','100'],stdout=monitor_log,stderr=subprocess.STDOUT)
  started=time.perf_counter()
  for sample in manifest['samples']:
   path=source/sample['file'];expected=sample['candidate_expected_prediction'];uploaded=upload(path)
   for name,arguments in [('pose',{'action':'recognize_by_video','video_path':uploaded}),('pose_recognize_by_photo',{'image_path':str(path)}),('pose',{'action':'recognize_by_url','url':f'http://127.0.0.1:{fileserver.server_port}/{path.name}'})]:
    t0=time.perf_counter();result=call(name,arguments);validate(result,expected);records.append({'file':sample['file'],'tool':name,'arguments_mode':arguments.get('action','alias'),'latency_ms':(time.perf_counter()-t0)*1000,'result':result,'expected_prediction':expected})
  def independent(sample):
   result=call('pose',{'action':'recognize_by_video','video_path':str(source/sample['file'])});validate(result,sample['candidate_expected_prediction']);return True
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:report['concurrent_requests_passed']=all(pool.map(independent,manifest['samples'][:2]))
  empty=out/'empty.mp4';empty.write_bytes(b'');bad=call('pose',{'action':'recognize_by_video','video_path':str(empty)});assert bad['ok'] is False and bad['count']==0 and bad['persons']==[];missing=call('pose',{'action':'recognize_by_video','video_path':str(out/'not-present.mp4')});assert missing['ok'] is False and missing['count']==0;report['invalid_input_explicitly_rejected']=True;report['valid_requests']=len(records);report['valid_request_failures']=0;report['duration_s']=time.perf_counter()-started
  report.update(ros_checks(call,validate,source,manifest,mcp_port,out))
  info=call('pose',{'action':'info'});assert info['backend']=='action314_onnx_cpu';report['backend_info']=info;report['passed']=True
 except Exception as error:report['passed']=False;report['error']=type(error).__name__+': '+str(error)
 finally:
  if monitor:
   monitor.terminate()
   try:monitor.wait(timeout=3)
   except subprocess.TimeoutExpired:monitor.kill();monitor.wait()
   monitor_log.close();samples=[int(x) for x in re.findall(r'GR3D_FREQ\s+(\d+)%',(out/'tegrastats.txt').read_text())];report['gpu_mean_percent']=sum(samples)/len(samples) if samples else None;report['gpu_max_percent']=max(samples) if samples else None;report['gpu_samples']=len(samples)
  fileserver.shutdown();fileserver.server_close()
  server.terminate()
  try:server.wait(timeout=15)
  except subprocess.TimeoutExpired:server.kill();server.wait()
  log.close();report['records']=records;(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
 if not report['passed']:raise SystemExit(1)

def ros_checks(call,validate,source,manifest,port,out):
 import cv2,rclpy
 from rclpy.executors import SingleThreadedExecutor
 from rclpy.qos import qos_profile_sensor_data
 from sensor_msgs.msg import CompressedImage
 from std_msgs.msg import String
 os.environ.update(ROS_DOMAIN_ID='74',ROS_LOCALHOST_ONLY='1',ROS_LOG_DIR=str(out/'ros-probe-logs'))
 rclpy.init();node=rclpy.create_node('action314_probe_'+str(port));executor=SingleThreadedExecutor();executor.add_node(node);running=threading.Event();running.set()
 def spin():
  while running.is_set():executor.spin_once(timeout_sec=.05)
 thread=threading.Thread(target=spin,daemon=True);thread.start();oneshot=[];stream=[];topic='/action314_probe_'+str(port)+'/camera';sub=node.create_subscription(String,'/perception/pose',lambda m:oneshot.append(json.loads(m.data)),10);streamsub=node.create_subscription(String,topic+'/poses',lambda m:stream.append(json.loads(m.data)),10);publisher=node.create_publisher(CompressedImage,topic,qos_profile_sensor_data)
 try:
  sample=manifest['samples'][0];response=call('pose',{'action':'recognize_by_video','video_path':str(source/sample['file'])});validate(response,sample['candidate_expected_prediction']);deadline=time.monotonic()+5
  while not oneshot and time.monotonic()<deadline:time.sleep(.05)
  assert oneshot,'One-shot result did not reach ROS2 /perception/pose';validate(oneshot[-1],sample['candidate_expected_prediction'])
  started=call('pose',{'action':'start','input_topic':topic,'instance_id':'submission-probe'});assert started['state']=='running' and started['output_topic']==topic+'/poses';deadline=time.monotonic()+5
  while publisher.get_subscription_count()==0 and time.monotonic()<deadline:time.sleep(.05)
  assert publisher.get_subscription_count()>0,'ROS2 camera subscription was not discovered'
  capture=cv2.VideoCapture(str(source/sample['file']))
  try:
   for index in range(8):
    ok,frame=capture.read();assert ok;ok,jpeg=cv2.imencode('.jpg',frame);assert ok;message=CompressedImage();message.format='jpeg';message.data=jpeg.tobytes();message.header.stamp.sec=1000+index//8;message.header.stamp.nanosec=(index%8)*125000000;publisher.publish(message);time.sleep(.14)
  finally:capture.release()
  deadline=time.monotonic()+10
  while not stream and time.monotonic()<deadline:time.sleep(.05)
  assert stream,'No action JSON published for ROS2 camera frames';validate(stream[-1]);info=call('pose',{'action':'info'});assert info['stream_errors']==0;assert call('pose',{'action':'reset'})['ok'];assert call('pose',{'action':'stop','instance_id':'submission-probe'})['state']=='idle'
  return {'ros_oneshot_publish_received':True,'ros_camera_publish_received':True,'ros_stream_errors':info['stream_errors'],'ros_stream_messages':len(stream),'stream_accuracy_claim':False}
 finally:
  call('pose',{'action':'stop'});running.clear();thread.join(timeout=3);executor.remove_node(node);node.destroy_node();executor.shutdown();rclpy.shutdown()

if __name__=='__main__':main()
