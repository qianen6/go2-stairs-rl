import hashlib
from pathlib import Path
import tempfile
import unittest


class AssetTests(unittest.TestCase):
    def test_referenced_physics_change_is_rejected(self):
        from go2_stairs.assets import verify_asset_bundle
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'top.usd').write_bytes(b'child=physics.usd')
            (root/'physics.usd').write_bytes(b'mass=10')
            manifest = {'files': [{'path': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                                   'bytes': p.stat().st_size} for p in root.glob('*.usd')]}
            result = verify_asset_bundle(root/'top.usd', root, manifest)
            self.assertEqual(len(result['sha256']), 64)
            self.assertEqual(len(result['files']), 2)
            (root/'physics.usd').write_bytes(b'mass=11')
            with self.assertRaisesRegex(ValueError, 'hash'):
                verify_asset_bundle(root/'top.usd', root, manifest)

    def test_unlisted_top_and_traversal_are_rejected(self):
        from go2_stairs.assets import verify_asset_bundle
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                verify_asset_bundle(Path(folder)/'unknown.usd', folder, {'files': []})
            with self.assertRaises(ValueError):
                verify_asset_bundle(Path(folder)/'top.usd', folder,
                                    {'files': [{'path': '../escape', 'bytes': 0, 'sha256': '0'*64}]})
