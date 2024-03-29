def unwrap(component):
    return component.python_func.__mammoth_wrapped__


class Env:
    def __init__(self, *args):
        for v in args:
            v = unwrap(v)
            setattr(self, v.__name__, v)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
