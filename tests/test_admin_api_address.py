import pathlib
import re
import subprocess
import unittest


class TestAdminApiAddress(unittest.TestCase):
    def test_current_origin_is_used(self):
        source = (pathlib.Path(__file__).resolve().parent.parent / "proxy.py").read_text(encoding="utf-8")
        match = re.search(r'<code id="apiAddress"[^>]*></code><script>(.*?)</script>', source)
        self.assertIsNotNone(match)
        for origin in ("https://example.test", "http://127.0.0.1:8000"):
            script = "const element={};const document={getElementById:()=>element};const window={location:{origin:process.argv[1]}};" + match.group(1) + ";process.stdout.write(element.textContent)"
            result = subprocess.run(["node", "-e", script, origin], check=True, capture_output=True, text=True)
            self.assertEqual(result.stdout, origin + "/v1")


if __name__ == "__main__":
    unittest.main()
