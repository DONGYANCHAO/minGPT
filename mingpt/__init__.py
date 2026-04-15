try:
    from ._version import __version__, __version_tuple__, version, version_tuple
except ImportError:
    __version__ = version = "0.0.1"
    __version_tuple__ = version_tuple = (0, 0, 1)
