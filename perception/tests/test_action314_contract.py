"""Public pose-tool contract checks; model inference is tested on actual videos."""
import json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from plugins.pose import PosePerceptionPlugin,TOOLS

class ContractTests(unittest.TestCase):
 def setUp(self):self.plugin=PosePerceptionPlugin({},'test',None)
 def test_all_canonical_labels_in_exact_order(self):
  expected=json.loads((Path(__file__).resolve().parents[1]/'plugins/action314/runtime.json').read_text())['labels'];actual=self.plugin.dispatch('list_actions',{})['activities'];self.assertEqual(actual,[{'name':r['activity'],'name_zh':r['activity_zh']} for r in expected]);self.assertEqual(len(actual),126)
 def test_start_info_stop_is_consistent(self):
  self.plugin._model=lambda:object();self.assertEqual(self.plugin.dispatch('start',{})['state'],'running');self.assertEqual(self.plugin.dispatch('info',{})['state'],'running');self.assertEqual(self.plugin.dispatch('stop',{})['state'],'idle');self.assertEqual(self.plugin.dispatch('info',{})['state'],'ready')
 def test_missing_input_is_explicit_error_not_none_label(self):
  result=self.plugin.dispatch('recognize_by_video',{});self.assertFalse(result['ok']);self.assertEqual(result['count'],0);self.assertEqual(result['persons'],[])
 def test_topic_contract_matches_published_format(self):
  self.assertEqual(TOOLS[0]['name'],'pose');self.assertEqual(TOOLS[0]['topic_out'][0]['format'],'data/json');self.assertEqual(TOOLS[0]['topic_in'][0]['format'],'image/jpeg')

if __name__=='__main__':unittest.main()
