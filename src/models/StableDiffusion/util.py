import diffusion

class SwitchBlockWrapper:
    def __init__(self, block):
        self.block = block

    def forward(self, x, context=None, t_emb=None):
        if isinstance(self.block, diffusion.AttentionBlock) and context is not None:
            return self.block(x, context)
        elif isinstance(self.block, diffusion.ResidualBlock) and t_emb is not None:
            return self.block(x, t_emb)
        return self.block(x)

    def __call__(self, x, context=None, t_emb=None):
        return self.forward(x, context, t_emb)

    def __getattr__(self, name: str):
        return getattr(self.block, name)
