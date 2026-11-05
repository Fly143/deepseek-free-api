import asyncio
import concurrent.futures
import threading
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import DsAccount, SecretBox
import app.config as config
from fastapi import HTTPException
from tests.test_api_auth import request
from app.identity import new_device_id
import proxy


class Response:
    status_code = 200
    headers = {}
    text = '{"code":0}'

    def __init__(self, value):
        self.value = value

    def json(self):
        return self.value


class IdentityTests(unittest.TestCase):
    def test_account_identity_is_unique_encrypted_and_roundtrips(self):
        ids = {new_device_id() for _ in range(20)}
        self.assertEqual(len(ids), 20)
        device_id = next(iter(ids))
        with tempfile.TemporaryDirectory() as folder:
            box = SecretBox(Path(folder) / 'config.json')
            account = DsAccount(account_label='person@example.com', login_type='email', device_id=device_id, headers={'user-agent': 'original'})
            stored = account.to_save_dict(box)
            self.assertNotEqual(stored['device_id'], device_id)
            self.assertNotIn('device_id', account.to_dict())
            restored = DsAccount.from_storage_dict(stored, box)
            self.assertEqual(restored.device_id, device_id)
            self.assertEqual(restored.headers['user-agent'], 'original')
            self.assertEqual(DsAccount.from_storage_dict({'account_label': 'old', 'login_type': 'email'}, box).device_id, '')

    def test_device_setting_requires_authentication_and_prevents_shared_ids(self):
        response = asyncio.run(request('/api/accounts/example/device-id', {'Content-Type': 'application/json'}, 'PUT', b'{"device_id":"device"}'))
        self.assertEqual(response, 401)
        first = DsAccount(account_label='first', login_type='email', device_id='old-id', device_source='generated')
        second = DsAccount(account_label='second', login_type='email', device_id='another-id')
        with patch.object(proxy.config_manager, 'accounts', [first, second]), patch.object(proxy.config_manager, 'save'):
            with self.assertRaises(HTTPException):
                asyncio.run(proxy.set_account_device_id('first', {'device_id': 'another-id'}))
            result = asyncio.run(proxy.set_account_device_id('first', {'device_id': 'user-owned-id'}))
            self.assertTrue(result['ok'])
            self.assertEqual(first.device_id, 'user-owned-id')
            self.assertEqual(first.device_source, 'provided')
            self.assertEqual(second.device_id, 'another-id')

    def test_failed_concurrent_logins_reuse_persisted_identity(self):
        barrier = threading.Barrier(2)
        calls = []

        def failed_post(url, **kwargs):
            calls.append(kwargs)
            barrier.wait(timeout=5)
            response = Response({'code': 1, 'msg': 'failed'})
            response.text = '{"code":1}'
            return response

        with tempfile.TemporaryDirectory() as folder:
            config_file = Path(folder) / 'config.json'
            with patch.object(config, 'CONFIG_FILE', config_file), patch.object(config, 'LEGACY_FILE', Path(folder) / 'token.json'):
                manager = config.ConfigManager()
                with patch.object(proxy, 'config_manager', manager), patch.object(proxy.cffi_requests, 'post', side_effect=failed_post):
                    def run_login():
                        return asyncio.run(proxy.deepseek_login({'login_type': 'email', 'email': 'user@example.com', 'password': 'password'}))
                    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                        results = list(pool.map(lambda _: run_login(), range(2)))
                self.assertTrue(all(not result['ok'] for result in results))
                self.assertEqual(calls[0]['json']['device_id'], calls[1]['json']['device_id'])
                self.assertEqual(calls[0]['headers'], calls[1]['headers'])
                device_id = calls[0]['json']['device_id']
                self.assertNotIn(device_id, config_file.read_text())
                restored = config.ConfigManager().get_account_by_label('user@example.com')
                self.assertEqual(restored.device_id, device_id)
                self.assertEqual(restored.device_source, 'generated')

    def test_relogin_reuses_account_identity_and_headers(self):
        account = DsAccount(account_label='person@example.com', login_type='email', _email='person@example.com', _password='password', device_id='existing-id', headers={'user-agent': 'real-client', 'x-rangers-id': 'stable', 'authorization': 'Bearer old'})
        calls = []

        def post(url, **kwargs):
            calls.append((url, kwargs))
            if url.endswith('/users/login'):
                return Response({'code': 0, 'data': {'biz_data': {'user': {'token': 'new-token'}}}})
            return Response({'code': 0, 'data': {'biz_data': {'chat_session': {'id': 'new-session'}}}})

        with patch.object(proxy.config_manager, 'get_account_by_label', return_value=account), patch.object(proxy.config_manager, 'update_account') as update, patch.object(proxy.cffi_requests, 'post', side_effect=post), patch.object(proxy, 'on_new_session'):
            result = proxy.relogin({'account': account.account_label, 'login_type': 'email', '_email': account._email, '_password': account._password})
        self.assertEqual(result['device_id'], 'existing-id')
        self.assertEqual(calls[0][1]['json']['device_id'], 'existing-id')
        self.assertEqual(calls[0][1]['headers']['user-agent'], 'real-client')
        self.assertEqual(calls[0][1]['headers']['x-rangers-id'], 'stable')
        self.assertNotIn('authorization', calls[0][1]['headers'])
        self.assertEqual(calls[1][1]['headers']['authorization'], 'Bearer new-token')
        self.assertEqual(update.call_args.kwargs['device_id'], 'existing-id')


if __name__ == '__main__':
    unittest.main()
