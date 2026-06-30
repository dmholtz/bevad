import json
import logging
from typing import Any, Dict, Optional


def write_dict(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_dict(filename):
    with open(filename) as f:
        data = json.loads(f.read())

    return data


class Configurator:
    def _configure_logger(self, verbose=0) -> None:
        """Configure the logger."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
        self.handler.setFormatter(formatter)
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.DEBUG if verbose else logging.WARNING)

    @staticmethod
    def _recursive_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively update dictionary `d` with values from dictionary `u`."""
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = Configurator._recursive_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def configure(self, config: Optional[Dict[str, Any]]) -> None:
        """Configure.

        Args:
        ----
            config (dict): Configuration parameters.

        """
        if config:
            self.config: Dict[str, Any] = self._recursive_update(self.config, config)

    @classmethod
    def default_config(cls):
        print("Call default_config from Configurator")
        return {}

    def _update_self_with_config(self, config):
        ## TODO: It would be better to annotate variables as configurable or something
        def_config = self.default_config()
        obj_dict = self.__dict__

        for key in def_config:
            obj_dict[key] = def_config[key]
            if key in config:
                obj_dict[key] = config[key]

    def close_logger(self):
        """Close and remove the logger handler."""
        if self.logger and self.handler:
            self.logger.removeHandler(self.handler)
            self.handler.close()
