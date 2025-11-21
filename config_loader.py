import json
import logging
from pathlib import Path
from typing import Any, Dict

CONFIG_FILE_NAME = "config.json"


def _resolve_config_path(custom_path: str | Path | None = None) -> Path:
    """
    返回配置文件的绝对路径。

    优先使用调用方提供的路径，否则与当前模块同级目录下的 config.json。
    """
    if custom_path:
        return Path(custom_path).expanduser().resolve()
    return (Path(__file__).resolve().parent / CONFIG_FILE_NAME).resolve()


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """
    读取 JSON 配置文件并返回字典。

    Args:
        config_path: 可选，指定配置文件路径。

    Raises:
        FileNotFoundError: 配置文件不存在。
        json.JSONDecodeError: JSON 内容格式错误。
    """
    path = _resolve_config_path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


try:
    CONFIG = load_config()
except FileNotFoundError:
    logging.error("未找到 config.json，请确认文件存在于项目根目录。")
    CONFIG = {}
except json.JSONDecodeError as exc:
    logging.error("config.json 解析失败: %s", exc)
    CONFIG = {}

