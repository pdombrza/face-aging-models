from typing import NamedTuple


class CycleGANLossLambdaParams(NamedTuple):
    lambda_identity: float = 5.0
    lambda_cycle: float = 10.0
    lambda_total: float = 0.5
