from dataclasses import dataclass
from logging import Logger
from pathlib import Path
from typing import Literal

T_BROWSER_TYPE = Literal[
    "chrome", "firefox", "opera", "edge", "brave", "chromium", "vivaldi", "safari"
]


@dataclass
class KeyIdPair:
    key: str
    kid: str


@dataclass
class State:
    caption_locale: str
    bearer_token: str | None
    concurrent_downloads: int
    load_from_file: bool
    save_to_file: bool
    course_url: str
    info: bool
    id_as_course_name: bool
    browser: T_BROWSER_TYPE | None
    use_continuous_lecture_numbers: bool
    log_level: int
    download_dir: Path
    logger: Logger
    keys: KeyIdPair | None
    batch: bool
    embed_subs: bool
