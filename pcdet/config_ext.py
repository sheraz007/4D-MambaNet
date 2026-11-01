import yaml

def python_eval_constructor(loader, node):
    value = eval(loader.construct_scalar(node))
    return value

def apply_yaml_ext():
    # Register for both loaders used in this codebase
    yaml.add_constructor(u"!python/eval", python_eval_constructor, Loader=yaml.FullLoader)
    yaml.add_constructor(u"!python/eval", python_eval_constructor, Loader=yaml.SafeLoader)