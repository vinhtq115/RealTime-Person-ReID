from .psta import PSTA
from .semi import SEMI


supported_models = {
    "PSTA": PSTA,
    "SEMI": SEMI,
}


def init_model(name, **kwargs):
    if name not in supported_models:
        raise ValueError(f"Unsupported model: {name}. Available: {supported_models.keys()}")

    model = supported_models[name](**kwargs)

    # TODO: load weights (if provided)

    return model
