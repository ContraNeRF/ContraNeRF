import logging
import os
from typing import IO, Any, Callable, Dict, List, Union

import yaml
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as _CfgNode


BASE_KEY = "_BASE_"


class CfgNode(_CfgNode):

    @classmethod
    def _open_cfg(cls, filename: str) -> Union[IO[str], IO[bytes]]:
        return g_pathmgr.open(filename, "r")

    @classmethod
    def load_yaml_with_base(
        cls, filename: str, allow_unsafe: bool = False
    ) -> Dict[str, Any]:
        with cls._open_cfg(filename) as f:
            try:
                cfg = yaml.safe_load(f)
            except yaml.constructor.ConstructorError:
                if not allow_unsafe:
                    raise
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Loading config {} with yaml.unsafe_load. Your machine may "
                    "be at risk if the file contains malicious content.".format(
                        filename
                    )
                )
                f.close()
                with cls._open_cfg(filename) as f:
                    cfg = yaml.unsafe_load(f)

        def merge_a_into_b(a: Dict[str, Any], b: Dict[str, Any]) -> None:
            # merge dict a into dict b. values in a will overwrite b.
            for k, v in a.items():
                if isinstance(v, dict) and k in b:
                    assert isinstance(
                        b[k], dict
                    ), "Cannot inherit key '{}' from base!".format(k)
                    merge_a_into_b(v, b[k])
                else:
                    b[k] = v

        def _load_with_base(base_cfg_file: str) -> Dict[str, Any]:
            if base_cfg_file.startswith("~"):
                base_cfg_file = os.path.expanduser(base_cfg_file)
            if not any(map(base_cfg_file.startswith, ["/", "https://", "http://"])):
                # the path to base cfg is relative to the config file itself.
                base_cfg_file = os.path.join(os.path.dirname(filename), base_cfg_file)
            return cls.load_yaml_with_base(base_cfg_file, allow_unsafe=allow_unsafe)

        if BASE_KEY in cfg:
            if isinstance(cfg[BASE_KEY], list):
                base_cfg: Dict[str, Any] = {}
                base_cfg_files = cfg[BASE_KEY]
                for base_cfg_file in base_cfg_files:
                    merge_a_into_b(_load_with_base(base_cfg_file), base_cfg)
            else:
                base_cfg_file = cfg[BASE_KEY]
                base_cfg = _load_with_base(base_cfg_file)
            del cfg[BASE_KEY]

            merge_a_into_b(cfg, base_cfg)
            return base_cfg
        return cfg

    def merge_from_file(self, cfg_filename: str, allow_unsafe: bool = False) -> None:
        loaded_cfg = self.load_yaml_with_base(cfg_filename, allow_unsafe=allow_unsafe)
        loaded_cfg = type(self)(loaded_cfg)
        self.merge_from_other_cfg(loaded_cfg)

    def merge_from_other_cfg(self, cfg_other: "CfgNode") -> Callable[[], None]:
        assert (
            BASE_KEY not in cfg_other
        ), "The reserved key '{}' can only be used in files!".format(BASE_KEY)
        return super().merge_from_other_cfg(cfg_other)

    def merge_from_list(self, cfg_list: List[str]) -> Callable[[], None]:
        keys = set(cfg_list[0::2])
        assert (
            BASE_KEY not in keys
        ), "The reserved key '{}' can only be used in files!".format(BASE_KEY)
        return super().merge_from_list(cfg_list)

    def __setattr__(self, name: str, val: Any) -> None:  # pyre-ignore
        if name.startswith("COMPUTED_"):
            if name in self:
                old_val = self[name]
                if old_val == val:
                    return
                raise KeyError(
                    "Computed attributed '{}' already exists "
                    "with a different value! old={}, new={}.".format(name, old_val, val)
                )
            self[name] = val
        else:
            super().__setattr__(name, val)
