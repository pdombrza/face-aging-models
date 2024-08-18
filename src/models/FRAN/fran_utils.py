from typing import NamedTuple


class FRANLossLambdaParams(NamedTuple):
    lambda_l1: float = 1.0
    lambda_lpips: float = 1.0
    lambda_adv: float = 0.05
    