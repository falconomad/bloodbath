import importlib
import os
import unittest


class SettingsTests(unittest.TestCase):
    def test_starting_capital_defaults_when_env_missing(self):
        os.environ.pop("STARTING_CAPITAL", None)
        os.environ.pop("TOP20_STARTING_CAPITAL", None)

        import src.settings as settings

        importlib.reload(settings)
        self.assertEqual(settings.STARTING_CAPITAL, 2000.0)

    def test_starting_capital_uses_new_env_value(self):
        os.environ["STARTING_CAPITAL"] = "750"
        os.environ.pop("TOP20_STARTING_CAPITAL", None)

        import src.settings as settings

        importlib.reload(settings)
        self.assertEqual(settings.STARTING_CAPITAL, 750.0)

    def test_starting_capital_falls_back_to_legacy_env_value(self):
        os.environ.pop("STARTING_CAPITAL", None)
        os.environ["TOP20_STARTING_CAPITAL"] = "820"

        import src.settings as settings

        importlib.reload(settings)
        self.assertEqual(settings.STARTING_CAPITAL, 820.0)

    def test_manual_check_ticker_normalizes_env_value(self):
        os.environ["MANUAL_CHECK_TICKER"] = "  u  "

        import src.settings as settings

        importlib.reload(settings)
        self.assertEqual(settings.MANUAL_CHECK_TICKER, "U")


if __name__ == "__main__":
    unittest.main()
