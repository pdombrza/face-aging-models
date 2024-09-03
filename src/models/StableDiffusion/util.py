import functools
import diffusion


def switch_forward(forward_func):
    @functools.wraps(forward_func)
    def wrapper(layer, x, context=None, time_emb=None):
        if isinstance(layer, diffusion.AttentionBlock):
            return forward_func(x, context)
        elif isinstance(layer, diffusion.ResidualBlock):
            return forward_func(x, time_emb)
        else:
            return forward_func(x)
    return wrapper
