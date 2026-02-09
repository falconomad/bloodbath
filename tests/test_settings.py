import importlib
import os
import unittest


class SettingsTests(unittest.TestCase):
    def test_top20_starting_capital_defaults_when_env_missing(self):
        os.environ.pop("TOP20_STARTING_CAPITAL", None)

        import src.settings as settings

        importlib.reload(settings)
        self.assertEqual(settings.TOP20_STARTING_CAPITAL, 2000.0)

    def test_top20_starting_capital_uses_env_value(self):
        os.environ["TOP20_STARTING_CAPITAL"] = "750"

        import src.settings as settings

        importlib.reload(settings)
        self.assertEqual(settings.TOP20_STARTING_CAPITAL, 750.0)


if __name__ == "__main__":
    unittest.main()
