__all__ = ["build_contranerf"]

from .contranerf import ContraNeRF


def build_contranerf(cfg):
    return ContraNeRF(cfg)
