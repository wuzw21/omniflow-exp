"""Check frozen research files without importing or running any experiment."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest = json.loads((ROOT / 'provenance/legacy_snapshot.json').read_text())
    errors = []
    for row in manifest['files']:
        p = ROOT / row['path']
        if not p.is_file():
            errors.append({'path': row['path'], 'error': 'missing'})
        elif hashlib.sha256(p.read_bytes()).hexdigest() != row['sha256']:
            errors.append({'path': row['path'], 'error': 'sha256 mismatch'})
    print(json.dumps({'files': len(manifest['files']), 'errors': errors,
                      'scope': 'archive integrity only; no model or device acceptance'}, indent=2))
    raise SystemExit(bool(errors))

if __name__ == '__main__':
    main()
