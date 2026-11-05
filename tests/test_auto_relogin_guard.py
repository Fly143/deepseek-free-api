import os
import unittest
from unittest.mock import patch

import proxy


class AutoReloginGuardTest(unittest.TestCase):
    def test_disabled_does_not_relogin(self):
        with patch.dict(os.environ, {"DEEPSEEK_AUTO_RELOGIN": "false"}), patch.object(proxy, "relogin") as relogin:
            self.assertIsNone(proxy.auto_relogin({"account_label": "test"}))
            relogin.assert_not_called()

    def test_enabled_relogs_in(self):
        with patch.dict(os.environ, {"DEEPSEEK_AUTO_RELOGIN": "true"}), patch.object(proxy, "relogin", return_value={"token": "test"}) as relogin:
            self.assertEqual(proxy.auto_relogin({"account_label": "test"}), {"token": "test"})
            relogin.assert_called_once()


if __name__ == "__main__":
    unittest.main()
