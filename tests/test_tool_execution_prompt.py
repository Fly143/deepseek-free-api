import unittest

from tool_call import convert_messages_for_deepseek
from tool_dsml import build_dsml_tool_prompt


class ToolExecutionPromptTests(unittest.TestCase):
    def test_rules_apply_with_and_without_passthrough(self):
        tools = [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}]
        messages = [{"role": "user", "content": "创建文件夹"}]
        for passthrough in (False, True):
            prompt = convert_messages_for_deepseek(messages, tools, passthrough=passthrough)
            self.assertIn("Tools are executed by the client", prompt)
            self.assertIn("wait for its result", prompt)
            self.assertIn("unless a tool result confirms it", prompt)
            self.assertIn("do not invent an outcome", prompt)
        self.assertIn("If no tool is needed, respond normally.", build_dsml_tool_prompt(tools))
        self.assertEqual(build_dsml_tool_prompt([]), "")
