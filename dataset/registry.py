import importlib


class Registry:
    def __init__(self):
        self.dataset = {}

    def register(self, id: str, **kwargs):
        if id in self.dataset:
            raise ValueError("Cannot add {}".format(id))
        dataset_cls = load(kwargs["entry_point"])
        self.dataset[id] = dataset_cls


def register(id, **kwargs):
    registry.register(id, **kwargs)


def load(name):
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


registry = Registry()
