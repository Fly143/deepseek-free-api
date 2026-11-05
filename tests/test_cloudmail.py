import json
import tempfile
import unittest
from pathlib import Path

from app.config import ConfigManager, SecretBox


class CloudMailConfigTests(unittest.TestCase):
    def test_secret_is_encrypted_at_rest(self):
        with tempfile.TemporaryDirectory() as d:
            manager = ConfigManager.__new__(ConfigManager)
            import threading
            manager.config_file = Path(d) / 'config.json'
            manager.box = SecretBox(manager.config_file)
            manager.lock = threading.RLock()
            manager.accounts = []
            manager._proxy_url = ''
            manager._passthrough = False
            manager._compression_mode = 'compress'
            manager._admin_password = 'admin'
            manager._mailcx_api_key = ''
            manager._mailcx_domain = ''
            manager.set_cloudmail_config('https://mail.test', 'secret-value', 'inbox.test', 'deepseek.com')
            stored = json.loads(manager.config_file.read_text())
            self.assertTrue(stored['cloudmail_token'].startswith('enc:v1:'))
            self.assertNotIn('secret-value', manager.config_file.read_text())
            manager.load()
            self.assertEqual(manager.get_cloudmail_config()['token'], 'secret-value')
