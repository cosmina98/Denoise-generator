import yaml
try:
    from easydict import EasyDict as edict
except Exception:
    class edict(dict):
        def __getattr__(self, name):
            try:
                value = self[name]
            except KeyError as exc:
                raise AttributeError(name) from exc
            if isinstance(value, dict) and not isinstance(value, edict):
                value = edict(value)
                self[name] = value
            return value

        def __setattr__(self, name, value):
            self[name] = value


def get_config(config, seed):
    config_dir = f'./config/{config}.yaml'
    config = edict(yaml.load(open(config_dir, 'r'), Loader=yaml.FullLoader))
    config.seed = seed

    return config
