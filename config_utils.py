import logging
import os
from functools import lru_cache
from typing import Any, Dict

import yaml

DEFAULT_LOGGING_CONFIG = {
    "name": "rgate",
    "level": "INFO",
    "format": "| %(levelname)s | %(name)s | %(message)s",
    "sample_template": "测试样本 {index}/{total}\n输入={input}\n正确答案={true}\n模型预测={pred}",
}


def load_config() -> Dict[str, Any]:
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if not os.path.exists(config_path):
        return {"logging": dict(DEFAULT_LOGGING_CONFIG)}
    with open(config_path, "r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}
    data.setdefault("logging", dict(DEFAULT_LOGGING_CONFIG))
    return data


def get_logging_config() -> Dict[str, Any]:
    config = load_config()
    logging_config = config.get("logging", {})
    merged_config = dict(DEFAULT_LOGGING_CONFIG)
    merged_config.update(logging_config)
    return merged_config


@lru_cache(maxsize=1)
def get_logger(name: str | None = None) -> logging.Logger:
    logging_config = get_logging_config()
    logger_name = name or logging_config["name"]
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(logging_config["format"])
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging_config["level"].upper())
    logger.propagate = False
    return logger
