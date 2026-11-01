import os
from pathlib import Path

import yaml
from easydict import EasyDict


def apply_env_data_overrides(config):
    """Remap dataset roots from MAMBANET4D_DATA or DATA_PATH without changing model logic.

    - DATA_PATH: if set, replaces DATA_CONFIG.DATA_PATH directly.
    - MAMBANET4D_DATA: treated as the parent of ``vod/`` and ``tj4d/`` folders.
      Relative cfg paths under data/vod and data/tj4d are rewritten under that root.
    """
    data_cfg = config.get('DATA_CONFIG', None)
    if data_cfg is None:
        return config

    env_direct = os.environ.get('DATA_PATH')
    env_root = os.environ.get('MAMBANET4D_DATA')
    if env_direct:
        data_cfg.DATA_PATH = env_direct
    elif env_root:
        root = Path(env_root)
        cur = Path(str(data_cfg.DATA_PATH))
        parts = cur.parts
        if 'vod' in parts:
            i = parts.index('vod')
            data_cfg.DATA_PATH = str(root.joinpath(*parts[i:]))
        elif 'tj4d' in parts or 'TJ4D' in parts:
            # Prefer data/tj4d layout
            if 'tj4d' in parts:
                i = parts.index('tj4d')
                data_cfg.DATA_PATH = str(root.joinpath(*parts[i:]))
            else:
                data_cfg.DATA_PATH = str(root / 'tj4d')
        if 'lidar_path' in data_cfg:
            # Trailing slash required: temporal loader does string concat, not Path join.
            data_cfg.lidar_path = str(root / 'vod' / 'lidar') + '/'
        if 'clip_path' in data_cfg:
            data_cfg.clip_path = str(root / 'vod' / 'radar_5frames' / 'clips') + '/'
    return config


def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            logger.info('----------- %s -----------' % (key))
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        logger.info('%s.%s: %s' % (pre, key, val))


def cfg_from_list(cfg_list, config):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = config
        for subkey in key_list[:-1]:
            assert subkey in d, 'NotFoundKey: %s' % subkey
            d = d[subkey]
        subkey = key_list[-1]
        # Optional subset size. Absent from dataset yamls so defaults stay unchanged.
        if subkey not in d and subkey == 'MAX_FRAMES':
            try:
                d[subkey] = int(literal_eval(v))
            except Exception:
                d[subkey] = int(v)
            continue
        assert subkey in d, 'NotFoundKey: %s' % subkey
        try:
            value = literal_eval(v)
        except:
            value = v

        if type(value) != type(d[subkey]) and isinstance(d[subkey], EasyDict):
            key_val_list = value.split(',')
            for src in key_val_list:
                cur_key, cur_val = src.split(':')
                val_type = type(d[subkey][cur_key])
                cur_val = val_type(cur_val)
                d[subkey][cur_key] = cur_val
        elif type(value) != type(d[subkey]) and isinstance(d[subkey], list):
            val_list = value.split(',')
            for k, x in enumerate(val_list):
                val_list[k] = type(d[subkey][0])(x)
            d[subkey] = val_list
        else:
            assert type(value) == type(d[subkey]), \
                'type {} does not match original type {}'.format(type(value), type(d[subkey]))
            d[subkey] = value


def _resolve_base_config_path(base_path: str, cur_cfg_file: str) -> str:
    """
    Resolve _BASE_CONFIG_ paths robustly.
    - If absolute: use it
    - Else: try relative to repo ROOT_DIR, then relative to current cfg file directory
    """
    base = Path(base_path)
    if base.is_absolute():
        return str(base)

    root_dir = (Path(__file__).resolve().parent / '../').resolve()
    cand1 = (root_dir / base).resolve()
    if cand1.exists():
        return str(cand1)

    # Many configs in this repo use paths like "cfgs/..." which are relative to tools/
    if len(base.parts) > 0 and base.parts[0] == 'cfgs':
        cand_tools = (root_dir / 'tools' / base).resolve()
        if cand_tools.exists():
            return str(cand_tools)

    cand2 = (Path(cur_cfg_file).resolve().parent / base).resolve()
    return str(cand2)


def merge_new_config(config, new_config, cur_cfg_file: str):
    # First, recursively merge base configs
    if '_BASE_CONFIG_' in new_config and new_config['_BASE_CONFIG_'] is not None:
        base_cfg_path = _resolve_base_config_path(str(new_config['_BASE_CONFIG_']), cur_cfg_file)
        with open(base_cfg_path, 'r') as f:
            try:
                base_yaml = yaml.load(f, Loader=yaml.FullLoader)
            except Exception:
                f.seek(0)
                base_yaml = yaml.safe_load(f)
        if base_yaml is None:
            raise RuntimeError(f'Failed to load base config: {base_cfg_path}')
        merge_new_config(config=config, new_config=EasyDict(base_yaml), cur_cfg_file=base_cfg_path)

    for key, val in new_config.items():
        if key == '_BASE_CONFIG_':
            continue
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config or not isinstance(config.get(key, None), EasyDict):
            config[key] = EasyDict()
        merge_new_config(config[key], val, cur_cfg_file=cur_cfg_file)

    return config


def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            #new_config = yaml.safe_load(f, Loader=yaml.FullLoader)
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.safe_load(f)

        merge_new_config(config=config, new_config=EasyDict(new_config), cur_cfg_file=cfg_file)

    apply_env_data_overrides(config)
    return config


cfg = EasyDict()
cfg.ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
cfg.LOCAL_RANK = 0
