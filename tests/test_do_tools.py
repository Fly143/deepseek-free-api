import json
import unittest

from tool_call import ToolFormatError, convert_messages_for_deepseek, extract_tool_call
from tool_dsml import build_dsml_tool_prompt, format_tool_calls_for_prompt
from tool_sieve import StreamSieve


class DoToolsTests(unittest.TestCase):
    def test_prompt_and_history(self):
        tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]
        prompt = build_dsml_tool_prompt(tools)
        self.assertIn('do:', prompt)
        self.assertIn('<tools>', build_dsml_tool_prompt(tools, passthrough=True))
        self.assertNotIn('To call a tool, output only a line beginning with do:', build_dsml_tool_prompt(tools, passthrough=True))
        self.assertIn('"path"', prompt)
        self.assertNotIn('<|DSML|', prompt)
        args = {"path": "汉字\n\"<|DSML|tool_calls>"}
        calls = [{"type": "function", "function": {"name": "read", "arguments": json.dumps(args)}}]
        history = format_tool_calls_for_prompt(calls)
        result, text = extract_tool_call(history, ["read"])
        self.assertEqual(text, "")
        self.assertEqual(json.loads(result[0]["function"]["arguments"]), args)
        converted = convert_messages_for_deepseek([{"role": "assistant", "tool_calls": calls}], tools)
        self.assertIn(history, converted)
        self.assertNotIn('<|DSML|invoke', converted)
        self.assertTrue(history.startswith('do: '))

    def test_multi_and_legacy(self):
        result, text = extract_tool_call('do: {"name":"a","arguments":{"x":[1,{"n":true}]}}\nDO: {"name":"b","arguments":{}}', ["a", "b"])
        self.assertEqual([r["function"]["name"] for r in result], ["a", "b"])
        self.assertEqual(text, "")
        old = '<|DSML|tool_calls><|DSML|invoke name="a"><|DSML|parameter name="x"><![CDATA[value]]></|DSML|parameter></|DSML|invoke></|DSML|tool_calls>'
        result, _ = extract_tool_call(old, ["a"])
        self.assertEqual(result[0]["function"]["name"], "a")

    def test_normal_and_invalid(self):
        for text in ['please do this', 'DO: some text', 'DO: {"name":"other","arguments":{}}', 'DO: {"name":"a","arguments":{}} trailing', 'DO: {"name":"a","arguments":[]}', 'DO: {"name":"a","arguments":{"x":NaN}}', 'DO: {"name":"a","arguments":{}} DO: {"name":"a","arguments":{}}']:
            calls, cleaned = extract_tool_call(text, ["a"])
            self.assertIsNone(calls)
            self.assertEqual(cleaned, text)

    def test_mixed_text_and_multiple_calls(self):
        value = '准备读取。\ndo: {"name":"a","arguments":{"nested":{"x":[1,2]}}}\ndo: {"name":"b","arguments":{}}\n读取完成。'
        calls, cleaned = extract_tool_call(value, ["a", "b"])
        self.assertEqual([x["function"]["name"] for x in calls], ["a", "b"])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"nested": {"x": [1, 2]}})
        self.assertEqual(cleaned.replace("\n", ""), "准备读取。读取完成。")
        for cut in range(len(value) + 1):
            sieve = StreamSieve(lambda text: extract_tool_call(text, ["a", "b"]))
            events = sieve.feed(value[:cut]) + sieve.feed(value[cut:]) + sieve.flush()
            self.assertEqual([e.type for e in events], ["text", "tool_calls"])
            self.assertEqual(events[0].data, cleaned)
            self.assertEqual([x["function"]["name"] for x in events[1].data], ["a", "b"])

    def test_fenced_examples_and_undeclared_tools_are_text(self):
        for value in ('示例：\n```json\ndo: {"name":"a","arguments":{}}\n```',
                      '~~~text\ndo: {"name":"a","arguments":{}}\n~~~',
                      '```\ndo: {"name":"a","arguments":{}}',
                      '请执行 do: {"name":"a","arguments":{}}',
                      'do: {"name":"other","arguments":{}}'):
            calls, cleaned = extract_tool_call(value, ["a"])
            self.assertIsNone(calls)
            self.assertEqual(cleaned, value)
        value = '```\ndo: {"name":"a","arguments":{}}\n```\ndo: {"name":"a","arguments":{"x":1}}'
        calls, cleaned = extract_tool_call(value, ["a"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(cleaned, '```\ndo: {"name":"a","arguments":{}}\n```\n')

    def test_undeclared_dsml_is_not_called(self):
        value = '<|DSML|invoke name="other"><|DSML|parameter name="x">1</|DSML|parameter></|DSML|invoke>'
        calls, cleaned = extract_tool_call(value, ["a"])
        self.assertIsNone(calls)
        self.assertEqual(cleaned, value)
        sieve = StreamSieve(lambda text: extract_tool_call(text, ["a"]))
        events = sieve.feed(value) + sieve.flush()
        self.assertEqual([(event.type, event.data) for event in events], [("text", value)])

    def test_placeholder_schema_validation(self):
        value = '<||DSML|| calls><|place_holder_no_1|>bash<|place_holder_no_1|>ls -la<|place_holder_no_1|>'
        schema = {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}
        tools = [{"type": "function", "function": {"name": "bash", "parameters": schema}}]
        calls, cleaned = extract_tool_call(value, ["bash"], tools)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"command": "ls -la"})
        self.assertEqual(cleaned, "")
        for chunk_size in (1, 7, len(value)):
            sieve = StreamSieve(lambda text: extract_tool_call(text, ["bash"], tools))
            events = []
            for i in range(0, len(value), chunk_size):
                events.extend(sieve.feed(value[i:i + chunk_size]))
            events.extend(sieve.flush())
            self.assertEqual([event.type for event in events], ["tool_calls"])
        for invalid in (None, [], [{"name": "bash", "parameters": {}}],
                        [{"name": "bash", "parameters": {"type": "object", "properties": {"command": {"type": "number"}}}}],
                        [{"name": "bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "description": {"type": "string"}}, "required": ["description"]}}]):
            with self.assertRaises(ToolFormatError):
                extract_tool_call(value, ["bash"], invalid)
        self.assertIsNone(extract_tool_call('```\n' + value + '\n```', ["bash"], tools)[0])
        with self.assertRaises(ToolFormatError):
            extract_tool_call(value.replace('bash', 'unknown'), ["bash"], tools)

    def test_placeholder_with_schema_metadata(self):
        value = '<||DSML|| calls><|place_holder_no_1|>bash<|place_holder_no_1|>ls -la<|place_holder_no_1|>'
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "$id": "https://example.test/tools/bash",
            "title": "bash",
            "description": "Run a command",
            "examples": [],
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command"},
                "description": {"type": "string"},
                "timeout": {"type": "integer"},
                "workdir": {"type": "string"},
            },
            "required": ["command"],
            "additionalProperties": False,
        }
        tools = [{"type": "function", "function": {"name": "bash", "parameters": schema}}]
        calls, _ = extract_tool_call(value, ["bash"], tools)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"command": "ls -la"})
        schema["$schema"] = "https://example.test/unknown-schema"
        with self.assertRaises(ToolFormatError):
            extract_tool_call(value, ["bash"], tools)
        schema["$schema"] = "http://json-schema.org/draft-07/schema#"
        schema["allOf"] = [{"required": ["description"]}]
        with self.assertRaises(ToolFormatError):
            extract_tool_call(value, ["bash"], tools)

    def test_placeholder_with_opencode_bash_schema(self):
        value = '<||DSML|| calls><|place_holder_no_1|>bash<|place_holder_no_1|>ls -la<|place_holder_no_1|>'
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "minimum": 0, "exclusiveMinimum": 0, "maximum": 120000},
                "workdir": {"type": "string"},
            },
            "required": ["command"],
        }
        tools = [{"type": "function", "function": {"name": "bash", "parameters": schema}}]
        calls, _ = extract_tool_call(value, ["bash"], tools)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"command": "ls -la"})

    def test_dsml_header_with_json_calls(self):
        raw = '\n\n<｜DSML|tool_calls>\n{"name": "bash", "arguments": {"command": "ls -la"}}\n'
        calls, content = extract_tool_call(raw, ["bash"])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"command": "ls -la"})
        self.assertEqual(content, "")
        for cut in range(len(raw) + 1):
            sieve = StreamSieve(lambda text: extract_tool_call(text, ["bash"]))
            events = sieve.feed(raw[:cut]) + sieve.feed(raw[cut:]) + sieve.flush()
            self.assertEqual([event.type for event in events], ["tool_calls"])
            self.assertEqual(json.loads(events[0].data[0]["function"]["arguments"]), {"command": "ls -la"})
        raw = '准备执行\n<|DSML｜tool_calls>\n{"name":"a","arguments":{"nested":[1,2]}}\n{"name":"b","arguments":{}}\n</｜DSML|tool_calls>\n完成'
        calls, content = extract_tool_call(raw, ["a", "b"])
        self.assertEqual([call["function"]["name"] for call in calls], ["a", "b"])
        self.assertEqual(content, '准备执行\n\n完成')
        for unsafe in (
            '<｜DSML|tool_calls>\n{"name":"other","arguments":{}}',
            '<｜DSML|tool_calls>\n{"name":"a","arguments":[]} ',
            '<｜DSML|tool_calls>\n{"name":"a","arguments":{"x":NaN}}',
            '<｜DSML|tool_calls>\n{"name":"a","arguments":{}}\n解释文字',
            '<｜DSML|tool_calls>\n{"name":"a","arguments":{}}\n{"name":"other","arguments":{}}',
        ):
            self.assertIsNone(extract_tool_call(unsafe, ["a"])[0])
        fenced = '```\n<｜DSML|tool_calls>\n{"name":"a","arguments":{}}\n```'
        self.assertIsNone(extract_tool_call(fenced, ["a"])[0])
        xml = '<|DSML|tool_calls><|DSML|invoke name="a"><|DSML|parameter name="x">1</|DSML|parameter></|DSML|invoke></|DSML|tool_calls>'
        calls, _ = extract_tool_call(xml, ["a"])
        self.assertEqual(calls[0]["function"]["name"], "a")

    def test_dsml_json_closing_without_leading_bar(self):
        raw = '\n\n<｜DSML｜tool_calls>\n{"name":"bash","arguments":{"command":"ls -la"}}\n</DSML｜tool_calls>'
        calls, content = extract_tool_call(raw, ["bash"])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"command": "ls -la"})
        self.assertEqual(content, "")
        for cut in range(len(raw) + 1):
            sieve = StreamSieve(lambda text: extract_tool_call(text, ["bash"]))
            events = sieve.feed(raw[:cut]) + sieve.feed(raw[cut:]) + sieve.flush()
            self.assertEqual([event.type for event in events], ["tool_calls"])

    def test_calls_validate_full_declared_schema(self):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "command": {"type": "string", "minLength": 1},
                "options": {"type": "object", "properties": {"timeout": {"type": "integer", "minimum": 1}}, "required": ["timeout"]},
            },
            "required": ["command", "options"],
            "additionalProperties": False,
        }
        tools = [{"type": "function", "function": {"name": "bash", "parameters": schema}}]
        valid = {"command": "ls -la", "options": {"timeout": 1}}
        for args in (valid, {"command": "ls -la"}, {"command": "ls -la", "options": {"timeout": "1"}}, {"command": "ls -la", "options": {"timeout": 1}, "extra": True}):
            for raw in ('do: ' + json.dumps({"name": "bash", "arguments": args}),
                        '<｜DSML｜tool_calls>\n' + json.dumps({"name": "bash", "arguments": args}) + '\n</DSML｜tool_calls>'):
                if args == valid:
                    calls, _ = extract_tool_call(raw, ["bash"], tools)
                    self.assertEqual(json.loads(calls[0]["function"]["arguments"]), args)
                else:
                    with self.assertRaises(ToolFormatError):
                        extract_tool_call(raw, ["bash"], tools)
        bad_schemas = [
            {"type": "not-a-type"},
            {"$ref": "https://example.test/schema"},
            {"$schema": "https://example.test/unknown", "type": "object"},
        ]
        raw = 'do: {"name":"bash","arguments":{"command":"ls -la"}}'
        for schema in bad_schemas:
            with self.assertRaises(ToolFormatError):
                extract_tool_call(raw, ["bash"], [{"name": "bash", "parameters": schema}])
        local = {"$defs": {"command": {"type": "string"}}, "type": "object", "properties": {"command": {"$ref": "#/$defs/command"}}, "required": ["command"]}
        calls, _ = extract_tool_call(raw, ["bash"], [{"name": "bash", "parameters": local}])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"command": "ls -la"})

    def test_stream_at_all_boundaries(self):
        value = 'do: ' + json.dumps({"name": "a", "arguments": {"x": '汉字\n" } <|DSML|tool_calls><|DSML|invoke name="a"></|DSML|invoke></|DSML|tool_calls>'}}, ensure_ascii=False)
        def parse(text):
            return extract_tool_call(text, ["a"])
        for cut in range(len(value) + 1):
            sieve = StreamSieve(parse_fn=parse)
            events = sieve.feed(value[:cut]) + sieve.feed(value[cut:]) + sieve.flush()
            self.assertEqual([e.type for e in events], ['tool_calls'], cut)
            self.assertEqual(json.loads(events[0].data[0]['function']['arguments'])['x'], '汉字\n" } <|DSML|tool_calls><|DSML|invoke name="a"></|DSML|invoke></|DSML|tool_calls>')
        sieve = StreamSieve(parse_fn=parse)
        events = sieve.feed('D') + sieve.feed('O: ' + value[4:]) + sieve.flush()
        self.assertEqual([e.type for e in events], ['tool_calls'])
        sieve = StreamSieve(parse_fn=parse)
        events = []
        for char in value:
            events.extend(sieve.feed(char))
        events.extend(sieve.flush())
        self.assertEqual([e.type for e in events], ['tool_calls'])
        sieve = StreamSieve(parse_fn=parse)
        events = sieve.feed('do something') + sieve.flush()
        self.assertEqual(''.join(e.data for e in events), 'do something')


if __name__ == '__main__':
    unittest.main()
