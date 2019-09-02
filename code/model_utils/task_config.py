import ast
import json
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging


class TaskConfig(object):
    def __init__(self, param_map=None, conf_file_path=None):
        self._param_map = {}
        try:
            config = json.load(file_io.FileIO(conf_file_path, 'r'))
            if not config:
                logging.error("config file not exists")
            if config['parameters']:
                self._param_map = config['parameters']
            if param_map:
                self._param_map.update(param_map)
        except:
            logging.info("load conf error!")

    def get_config(self, config_name, default=None):
        return self._param_map.get(config_name, default)

    def get_config_as_int(self, config_name, default=None):
        value_str = self.get_config(config_name, default)
        return int(value_str) if value_str else value_str

    def get_config_as_float(self, config_name, default=None):
        value_str = self.get_config(config_name, default)
        return float(value_str) if value_str else value_str

    def get_config_as_bool(self, config_name, default=None):
        raw_value = self.get_config(config_name, default)
        if raw_value and isinstance(raw_value, bool):
            return raw_value
        elif raw_value and (isinstance(raw_value, str) or isinstance(raw_value, unicode)):
            return ast.literal_eval(raw_value)
        else:
            return False

    def get_config_as_list(self, config_name, default=None):
        raw_value = self.get_config(config_name, default)
        if raw_value and isinstance(raw_value, list):
            return raw_value
        else:
            return ast.literal_eval(raw_value)

    def contains(self, config_name):
        return config_name in self._param_map

    def add_config(self, key, value):
        self._param_map[key] = value

    def add_if_not_contain(self, key, value):
        if not self.contains(key):
            self.add_config(key, value)
