import unittest

from proxy import ADMIN, app


class RegistrationRoutesTests(unittest.TestCase):
    def test_registration_routes_and_ui_are_removed(self):
        paths = set(app.openapi()["paths"])
        self.assertFalse(any(path.startswith("/api/register") for path in paths))
        self.assertNotIn("registerPanel", ADMIN)
        self.assertNotIn("switchTab('register')", ADMIN)
        self.assertIn("/api/login", paths)
        self.assertIn("/api/config", paths)
        self.assertIn("/v1/chat/completions", paths)


if __name__ == '__main__':
    unittest.main()
