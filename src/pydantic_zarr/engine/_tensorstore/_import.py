TENSORSTORE_INSTALLED = False

try:
    import tensorstore

    TENSORSTORE_INSTALLED = True
except ImportError:
    pass
