"""CPU-only 2D evidence. No training or ground-truth dependencies."""

__version__ = "0.1.0"


def analyze(*args, **kwargs):
    from .pipeline import analyze as run

    return run(*args, **kwargs)


def get_region(*args, **kwargs):
    from .pipeline import get_region as run

    return run(*args, **kwargs)
