"""Exercise the real model installation contract, without trained data."""
import functools,hashlib,http.server,json,os,sys,tarfile,tempfile,threading,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from plugins.action314 import model_store

class ModelStoreTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.here=self.root/'source';self.here.mkdir();self.cache=self.root/'cache';self.payload=b'known-test-payload';(self.here/'joint.onnx').write_bytes(self.payload);self.archive=self.here/'models.tar.gz'
  with tarfile.open(self.archive,'w:gz') as archive:archive.add(self.here/'joint.onnx',arcname='joint.onnx')
  self.server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(http.server.SimpleHTTPRequestHandler,directory=str(self.here)));threading.Thread(target=self.server.serve_forever,daemon=True).start();self.spec={'url':f'http://127.0.0.1:{self.server.server_port}/models.tar.gz','archive_sha256':model_store.digest(self.archive),'max_download_bytes':1024*1024,'files':{'joint.onnx':hashlib.sha256(self.payload).hexdigest()}};self._save()
 def _save(self):(self.here/'asset.json').write_text(json.dumps(self.spec))
 def tearDown(self):self.server.shutdown();self.server.server_close();self.temp.cleanup()
 def _install(self,offline='0'):
  with patch.object(model_store,'HERE',self.here),patch.dict(os.environ,{'ACTION314_MODEL_DIR':str(self.cache),'ACTION314_OFFLINE':offline}):return model_store.ensure_models()
 def test_download_verify_and_reuse_offline(self):
  self.assertEqual((self._install()/'joint.onnx').read_bytes(),self.payload);self.assertEqual(self._install('1'),self.cache)
 def test_corruption_cannot_pass_offline(self):
  self._install();(self.cache/'joint.onnx').write_bytes(b'corrupt')
  with self.assertRaises(RuntimeError):self._install('1')
 def test_wrong_archive_digest_is_rejected(self):
  self.spec['archive_sha256']='0'*64;self._save()
  with self.assertRaises(ValueError):self._install()
  self.assertFalse((self.cache/'joint.onnx').exists())
 def test_extra_archive_member_is_rejected(self):
  with tarfile.open(self.archive,'w:gz') as archive:archive.add(self.here/'joint.onnx',arcname='joint.onnx');archive.add(self.here/'joint.onnx',arcname='../outside.onnx')
  self.spec['archive_sha256']=model_store.digest(self.archive);self._save()
  with self.assertRaises(ValueError):self._install()
  self.assertFalse((self.root/'outside.onnx').exists())
 def test_member_digest_is_rejected(self):
  self.spec['files']['joint.onnx']='0'*64;self._save()
  with self.assertRaises(ValueError):self._install()

if __name__=='__main__':unittest.main()
