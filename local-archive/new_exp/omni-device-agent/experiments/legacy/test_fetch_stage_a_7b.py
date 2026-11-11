"""Regression for a mirror's redirect body being larger than its tiny JSON file."""
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from fetch_stage_a_7b import fetch_model,valid


class MirrorFetchTests(unittest.TestCase):
    def test_large_redirect_body_does_not_reject_valid_small_config(self):
        payload=b'{"key":1}\n'
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.startswith('/repo/resolve/'):
                    body=b'x'*2000
                    self.send_response(302)
                    self.send_header('Location','/actual')
                else:
                    body=payload
                    self.send_response(200)
                self.send_header('Content-Length',str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self,*args):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        entry={'name':'config.json','bytes':len(payload),'sha256':None,
               'git_blob_sha1':hashlib.sha1(f'blob {len(payload)}\0'.encode()+payload).hexdigest()}
        model={'repo':'repo','revision':'fixed','files':[entry]}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                dest=Path(fetch_model(Path(tmp),'base',model,f'http://127.0.0.1:{server.server_port}'))
                self.assertEqual((dest/'config.json').read_bytes(),payload)
                self.assertTrue(valid(dest/'config.json',entry))
                (dest/'config.json').write_bytes(b'x'*len(payload))
                self.assertFalse(valid(dest/'config.json',entry))
        finally:
            server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
