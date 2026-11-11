"""Verify every file in the locked Go2 USD dependency bundle, not just its root."""
import hashlib
import json
from pathlib import Path


def verify_asset_bundle(usd_path, root, manifest):
    root = Path(root).resolve()
    top = Path(usd_path).resolve()
    records = manifest.get('files', [])
    if not records:
        raise ValueError('asset manifest is empty')
    files = []
    paths = set()
    for record in records:
        path = (root/record['path']).resolve()
        if not path.is_relative_to(root) or path in paths:
            raise ValueError('asset path escapes root or is duplicated')
        paths.add(path)
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != record['sha256'] or len(content) != record['bytes']:
            raise ValueError(f'asset hash/size mismatch: {path}')
        files.append({'path': path.relative_to(root).as_posix(), 'bytes': len(content), 'sha256': digest})
    if top not in paths:
        raise ValueError('top USD is not in the locked asset bundle')
    files.sort(key=lambda r: r['path'])
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'sha256': digest, 'files': files, 'root': str(root), 'usd_path': str(top)}


def locked_go2_assets(usd_path):
    top = Path(usd_path).resolve()
    # The provisioned Go2 bundle has four files; hashes bind its known dependency closure.
    lock = Path(__file__).resolve().parents[2]/'configs/runtime-lock.json'
    manifest = json.loads(lock.read_text(encoding='utf-8'))['asset_manifest']
    return verify_asset_bundle(top, top.parents[2], manifest)
