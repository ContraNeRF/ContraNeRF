__all__ = ["build_model"]

from .model import build_contranerf

model_mapper = {
    "ContraNeRF": build_contranerf
}


def build_model(cfg):
    name = cfg.model.name
    model = model_mapper[name](cfg)
    return model
