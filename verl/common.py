import importlib


def import_and_get(class_path: str):
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name, package=None)
    return getattr(module, class_name)


def make_object(class_path: str, args=None):
    _class = import_and_get(class_path)
    return _class() if args is None else _class(**args)


def make_object_from_config(config):
    if isinstance(config, list):
        return [make_object_from_config(i) for i in config]
    elif isinstance(config, dict):
        if '_function_name' in config:
            return import_and_get(config['_function_name'])
        elif '_class_name' in config:
            return make_object(
                config['_class_name'],
                {
                    k: make_object_from_config(v)
                    for k, v in config.items()
                    if k != '_class_name'
                }
            )
        else:
            return {
                k: make_object_from_config(v)
                for k, v in config.items()
            }
    else:
        return config
