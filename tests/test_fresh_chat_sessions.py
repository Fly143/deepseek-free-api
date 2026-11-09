import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import app.config as config
import app.lock_guard as guard
import context_compressor
import proxy
import test_do_api_integration as integration


class FreshChatSessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        with patch.object(config, 'CONFIG_FILE', Path(self.tmp.name) / 'config.json'), patch.object(config, 'LEGACY_FILE', Path(self.tmp.name) / 'token.json'):
            self.manager = config.ConfigManager()
        self.manager.add_account(config.DsAccount(account_label='test', login_type='email', token='token', session_id='old', is_valid=True))
        self.patches = [
            patch.object(proxy, 'config_manager', self.manager),
            patch.object(proxy, 'get_pow_response', return_value=''),
            patch.object(proxy, 'get_models', return_value={'deepseek-default': (False, False, 64000, 8000)}),
            patch.object(guard, 'MIN_INTERVAL_SEC', 0),
            patch.object(guard, '_slots', threading.Semaphore(1)),
            patch.dict(os.environ, {'DEEPSEEK_PROXY_API_KEY': 'test-key'}),
        ]
        for item in self.patches:
            item.start()

    async def asyncTearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.tmp.cleanup()

    def upstream(self, fail=False):
        created = []
        completions = []

        def post(url, **kwargs):
            if url.endswith('/chat_session/create'):
                response = integration.UpstreamResponse([])
                if fail:
                    response.status_code = 500
                    response.json = lambda: {}
                else:
                    session_id = f'fresh-{len(created) + 1}'
                    created.append(session_id)
                    response.json = lambda: {'data': {'biz_data': {'chat_session': {'id': session_id}}}}
                return response
            if url.endswith('/chat/completion'):
                completions.append(kwargs['json'])
                return integration.UpstreamResponse(['ok'])
            raise AssertionError(url)

        return post, created, completions

    async def test_each_request_uses_unique_session_and_forwards_history(self):
        post, created, completions = self.upstream()
        messages = [{'role': 'user', 'content': '先前问题'}, {'role': 'assistant', 'content': '先前回答'}, {'role': 'user', 'content': '现在的问题'}]
        with patch.object(proxy.cffi_requests, 'post', side_effect=post), patch.object(proxy, 'on_new_session'), patch.object(proxy, 'add_tokens'):
            for stream in (False, True):
                status, data = await integration.request({'model': 'deepseek-default', 'messages': messages, 'stream': stream})
                self.assertEqual(status, 200, data)
        self.assertEqual(len(created), 2)
        self.assertEqual([item['chat_session_id'] for item in completions], created)
        for item in completions:
            for message in messages:
                self.assertIn(message['content'], item['prompt'])

    async def test_session_creation_failure_does_not_use_old_session(self):
        post, _, completions = self.upstream(fail=True)
        with patch.object(proxy.cffi_requests, 'post', side_effect=post):
            status, _ = await integration.request({'model': 'deepseek-default', 'messages': [{'role': 'user', 'content': 'hello'}]})
        self.assertGreaterEqual(status, 400)
        self.assertEqual(completions, [])

    async def test_vision_upload_and_completion_use_fresh_session(self):
        post, created, completions = self.upstream()
        uploads = []
        def upload(data, filename, content_type, cfg):
            uploads.append(cfg['session_id'])
            return 'file-1'
        models = {'deepseek-vision': (False, False, 64000, 8000), 'deepseek-default': (False, False, 64000, 8000)}
        body = {'model': 'deepseek-vision', 'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': '描述图片'}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,aGVsbG8='}}]}]}
        with patch.object(proxy.cffi_requests, 'post', side_effect=post), patch.object(proxy, 'get_models', return_value=models), patch.object(proxy, 'upload_file_to_deepseek', side_effect=upload), patch.object(proxy, 'fork_file_to_vision', return_value='vision-1'), patch.object(proxy, 'wait_for_file_parsing', return_value=['vision-1']), patch.object(proxy, 'on_new_session'), patch.object(proxy, 'add_tokens'):
            status, data = await integration.request(body)
        self.assertEqual(status, 200, data)
        self.assertEqual(len(created), 1)
        self.assertEqual(uploads, created)
        self.assertEqual(completions[0]['chat_session_id'], created[0])
        self.assertIn('vision-1', completions[0]['ref_file_ids'])

    async def test_summary_and_final_use_separate_sessions(self):
        post, created, completions = self.upstream()
        async def compress(messages, model, summarize):
            await summarize('请总结之前的内容', model)
            return 'summary', [{'role': 'user', 'content': '总结后的上下文'}]
        with patch.object(proxy.cffi_requests, 'post', side_effect=post), patch.object(self.manager, 'get_compression_mode', return_value='compress'), patch.object(context_compressor, 'should_compress', return_value=True), patch.object(context_compressor, 'compress_messages', side_effect=compress), patch.object(proxy, 'on_new_session'), patch.object(proxy, 'add_tokens'):
            status, data = await integration.request({'model': 'deepseek-default', 'messages': [{'role': 'user', 'content': 'hello'}]})
        self.assertEqual(status, 200, data)
        self.assertEqual(len(created), 2)
        self.assertEqual(set(item['chat_session_id'] for item in completions), set(created))
        self.assertEqual(len(completions), 2)
        self.assertIn('总结后的上下文', completions[-1]['prompt'])
