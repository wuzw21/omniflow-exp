"""One-shot launch gates on temporary files; not CUDA or model evidence."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from queue_stage_a_pilot import main


class QueueGateTests(unittest.TestCase):
    def fixture(self,root,verified=False,hashes=None):
        (root/'results').mkdir()
        model={'revision':'fixture-only-no-weights'}
        (root/'stage_a_7b_models.json').write_text(json.dumps({'models':{'fixture':model}}))
        if verified:
            folder=root/'models'/model['revision'];folder.mkdir(parents=True)
            (folder/'verified_manifest.json').write_text(json.dumps(model))
        h=root/'hashes.json';h.write_text(json.dumps(hashes or {}))
        return argparse.Namespace(root=str(root),installer_pid=999999999,source_hashes=str(h),wait_seconds=0)

    def invoke_without_subprocess(self,args,error):
        with patch('queue_stage_a_pilot.subprocess.run') as run,patch('queue_stage_a_pilot.subprocess.Popen') as popen:
            with redirect_stdout(io.StringIO()),self.assertRaises(error):main(args)
            run.assert_not_called();popen.assert_not_called()

    def test_incomplete_preparation_cannot_launch_a_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root)
            self.invoke_without_subprocess(args,TimeoutError)
            state=json.loads((root/'results/queued_pilot_state.json').read_text())
            self.assertEqual(state['children'],[])
            self.assertEqual(state['verified_models'],[])

    def test_existing_result_is_preserved_and_not_restarted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root,True)
            result=root/'results/stage_a_7b_pilot.json';result.write_text('previous partial result')
            self.invoke_without_subprocess(args,FileExistsError)
            self.assertEqual(result.read_text(),'previous partial result')

    def test_changed_source_stops_before_gpu_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root,True,{'source.py':'0'*64})
            (root/'source.py').write_text('changed')
            self.invoke_without_subprocess(args,ValueError)

    def test_existing_queue_record_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=self.fixture(root)
            record=root/'results/queued_pilot_state.json';record.write_text('existing job handle')
            self.invoke_without_subprocess(args,FileExistsError)
            self.assertEqual(record.read_text(),'existing job handle')


if __name__=='__main__':unittest.main()
