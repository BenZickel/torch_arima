import torch as pt

class JitEnabledProxy():
    def __init__(self):
        self.enabled = True

    def __bool__(self):
        return self.enabled

def disable_jit():
    _jit_enabled.enabled = False

def enable_jit():
    _jit_enabled.enabled = True

def jit_script(func):
    '''
    Decorator that decides whether to execute a JIT compiled version
    of func or func itself, at each call to the decorated function,
    based on the _jit_enabled flag.
    '''
    jit_script_func = pt.jit.script(func)
    def dynamic_jit_enabled_func(*args, **kwargs):
        if _jit_enabled:
            return jit_script_func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return dynamic_jit_enabled_func

_jit_enabled = JitEnabledProxy()
