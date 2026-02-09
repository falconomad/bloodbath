import os


def _float_from_env(name, default):
    raw = os.getenv(name)
    if raw is None:
        return float(default)

    try:
        return float(raw)
    except ValueError:
        return float(default)


TOP20_STARTING_CAPITAL = _float_from_env("TOP20_STARTING_CAPITAL", 2000)
