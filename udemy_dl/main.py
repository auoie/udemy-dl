from base64 import b64encode
import functools
from glob import glob
from http.cookiejar import CookieJar
import json
import logging
from logging import Logger
import operator
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import List, Literal, Optional, Tuple, assert_never
from bs4 import BeautifulSoup, NavigableString, Tag
import dotenv
import m3u8
from pydantic import BaseModel
import requests
import requests.cookies
from pathvalidate import sanitize_filename
from slugify import slugify
from typed_argparse import TypedArgs
import typed_argparse
from coloredlogs import ColoredFormatter
import browser_cookie3
from urllib.parse import urlparse
from requests.cookies import RequestsCookieJar

from udemy_dl import constants
from udemy_dl.state import T_BROWSER_TYPE, KeyIdPair, State
from udemy_dl.tls import SSLCiphers
from udemy_dl.download import (
    SegmentDL,
    Task,
    download_cloudflare_files,
    download_mp4,
    download_segment_stream,
)
from udemy_dl.generated import (
    extract_course_info_json,
    extract_course_json,
    filtered_lectures,
    filtered_chapters,
    filtered_quizzes,
    filtered_practices,
)


T_LOG_LEVEL = Literal["DEBUG", "INFO", "ERROR", "WARNING", "CRITICAL"]


class Arguments(TypedArgs):

    course_url: str = typed_argparse.arg("-c", help="The URL of the course to download")

    bearer_token: Optional[str] = typed_argparse.arg(
        "-b", help="The Bearer token to use"
    )

    lang: str = typed_argparse.arg(
        "-l",
        default="en",
        help="The language to download for captions, specify 'all' to download all captions",
    )

    concurrent_downloads: int = typed_argparse.arg(
        default=16,
        help="The number of maximum concurrent downloads for batch downloads and segments (HLS and DASH, must be a number 1-30)",
    )

    batch_playlists: bool = typed_argparse.arg(
        help="Batch download master playlists. Then batch download index playlists. Then sequentially download all other assets as normal."
    )

    embed_subs: bool = typed_argparse.arg(
        help="If a video has any subs, embed those subs in the output video"
    )

    decrypt: bool = typed_argparse.arg(
        help="Decrypt the encrypted videos. Requires a {key: kid} pair in `keyfile.json`"
    )

    info: bool = typed_argparse.arg(
        help="If specified, only course information will be printed, nothing will be downloaded",
    )

    id_as_course_name: bool = typed_argparse.arg(
        help="If specified, the course id will be used in place of the course name for the output directory. This is a 'hack' to reduce the path length",
    )

    save_to_file: bool = typed_argparse.arg(
        help="If specified, course content will be saved to a file that can be loaded later with --load-from-file, this can reduce processing time (Note that asset links expire after a certain amount of time)",
    )

    load_from_file: bool = typed_argparse.arg(
        help="If specified, course content will be loaded from a previously saved file with --save-to-file, this can reduce processing time (Note that asset links expire after a certain amount of time)",
    )

    log_level: T_LOG_LEVEL = typed_argparse.arg(
        default="INFO",
        help="Logging level: one of DEBUG, INFO, ERROR, WARNING, CRITICAL",
    )

    browser: Optional[T_BROWSER_TYPE] = typed_argparse.arg(
        help="The browser to extract cookies from"
    )

    out: str = typed_argparse.arg(
        "-o", default="out_dir", help="Set the path to the output directory"
    )

    use_continuous_lecture_numbers: bool = typed_argparse.arg(
        help="Use continuous lecture numbering instead of per-chapter"
    )


def log_str_to_level(log_str: T_LOG_LEVEL) -> int:
    match log_str:
        case "CRITICAL":
            return logging.CRITICAL
        case "WARNING":
            return logging.WARNING
        case "ERROR":
            return logging.ERROR
        case "DEBUG":
            return logging.DEBUG
        case "INFO":
            return logging.INFO


def get_concurrent_downloads(cd: int) -> int:
    if cd < 1:
        return 10
    if cd > 30:
        return 30
    return cd


def get_keyidpair(keys: dict) -> KeyIdPair | None:
    key_list = list(keys.keys())
    if len(key_list) != 1:
        return None
    first_key = key_list[0]
    if not isinstance(first_key, str):
        return None
    value = keys[first_key]
    if not isinstance(value, str):
        return None
    return KeyIdPair(key=value, kid=first_key)


def create_logger(
    log_level: int,
    log_fmt: str,
    date_fmt: str,
    log_file_path: Path,
) -> Logger:
    # setup a logger
    logger = logging.getLogger(__name__)
    logging.root.setLevel(log_level)

    # create a colored formatter for the console
    console_formatter = ColoredFormatter(fmt=log_fmt, datefmt=date_fmt)
    # create a regular non-colored formatter for the log file
    file_formatter = logging.Formatter(fmt=log_fmt, datefmt=date_fmt)

    # create a handler for console logging
    stream = logging.StreamHandler()
    stream.setLevel(log_level)
    stream.setFormatter(console_formatter)

    # create a handler for file logging
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(file_formatter)

    # construct the logger
    logger = logging.getLogger("udemy-downloader")
    logger.setLevel(log_level)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def get_keys(key_file_path: Path) -> KeyIdPair | None:
    keys = None
    if key_file_path.exists():
        key_dict: dict = json.loads(key_file_path.read_bytes())
        keys = get_keyidpair(key_dict)
    return keys


def normalize_course_url(url: str, logger: Logger) -> str:
    result = urlparse(url)
    path_split = result.path.split("/")
    if len(path_split) <= 2:
        logger.fatal(f"url is invalid: {url}")
    course_name = path_split[2]
    return f"https://www.udemy.com/course/{course_name}/learn"


def get_state(args: Arguments):
    constants.LOG_DIR_PATH.mkdir(parents=True, exist_ok=True)
    log_level: int = log_str_to_level(args.log_level)
    download_dir = Path(args.out).resolve()
    logger: Logger = create_logger(
        log_level=log_level,
        log_fmt=constants.LOG_FORMAT,
        date_fmt=constants.LOG_DATE_FORMAT,
        log_file_path=constants.LOG_FILE_PATH,
    )
    logger.info(f"Output directory set to {download_dir}")
    download_dir.mkdir(parents=True, exist_ok=True)
    Path(constants.SAVED_DIR).mkdir(parents=True, exist_ok=True)
    keys: KeyIdPair | None = None
    if args.decrypt:
        keys = get_keys(constants.KEY_FILE_PATH)
        if keys is None:
            logger.error(
                "> Keyfile (or keys) not found! You won't be able to decrypt videos!"
            )
            sys.exit(1)
    return State(
        caption_locale=args.lang,
        bearer_token=args.bearer_token,
        concurrent_downloads=get_concurrent_downloads(args.concurrent_downloads),
        load_from_file=args.load_from_file,
        save_to_file=args.save_to_file,
        course_url=normalize_course_url(args.course_url, logger),
        info=args.info,
        id_as_course_name=args.id_as_course_name,
        browser=args.browser,
        use_continuous_lecture_numbers=args.use_continuous_lecture_numbers,
        download_dir=download_dir,
        keys=keys,
        log_level=log_level,
        logger=logger,
        batch=args.batch_playlists,
        embed_subs=args.embed_subs,
    )


def cookiejar_to_requestscookiejar(cookieJar: CookieJar) -> RequestsCookieJar:
    result = requests.cookies.RequestsCookieJar()
    requests.cookies.merge_cookies(result, cookieJar)
    return result


class Session(object):
    _session: requests.Session
    _headers: dict[str, str | None]
    _cj: RequestsCookieJar | None

    def __init__(self, cookieJar: CookieJar | None):
        self._cj = cookiejar_to_requestscookiejar(cookieJar) if cookieJar else None
        self._headers: dict[str, str | None] = {
            "Origin": "www.udemy.com",
            # "User-Agent":
            # "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0",
            "Accept": "*/*",
            "Accept-Encoding": None,
        }
        self._session = requests.sessions.Session()
        self._session.mount(
            "https://",
            SSLCiphers(
                cipher_list="ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-SHA384:ECDHE-ECDSA-AES256-SHA384:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-SHA256:AES256-SH"
            ),
        )

    def _set_auth_headers(self, bearer_token: str = ""):
        self._headers["Authorization"] = "Bearer {}".format(bearer_token)
        self._headers["X-Udemy-Authorization"] = "Bearer {}".format(bearer_token)

    def _get(self, url: str, logger: Logger) -> requests.Response | None:
        for i in range(10):
            response = self._session.get(url, headers=self._headers, cookies=self._cj)
            if response.ok:
                return response
            else:
                logger.error("Failed request " + url)
                logger.error(
                    f"{response.status_code} {response.reason}, retrying (attempt {i} )..."
                )
                time.sleep(0.8)
        return None

    def _get_allow_5xx(self, url: str, logger: Logger) -> requests.Response | None:
        for i in range(10):
            response = self._session.get(url, headers=self._headers, cookies=self._cj)
            if response.ok or response.status_code in [502, 503]:
                return response
            if not response.ok:
                logger.error("Failed request " + url)
                logger.error(
                    f"{response.status_code} {response.reason}, retrying (attempt {i} )..."
                )
                time.sleep(0.8)
        return None

    def terminate(self):
        self._set_auth_headers()
        return


class UdemyAuth(object):
    def __init__(self, session: Session):
        self._session = session

    def authenticate(self, bearer_token: str):
        self._session._set_auth_headers(bearer_token=bearer_token)
        return self._session


def get_session_and_cookies(
    auth: UdemyAuth, logger: Logger, bearer_token: str, browser: T_BROWSER_TYPE | None
):
    session = auth.authenticate(bearer_token=bearer_token)
    if not session:
        if browser is None:
            logger.error(
                "No bearer token was provided, and no browser for cookie extraction was specified."
            )
            sys.exit(1)


def browser_to_cookie(browser: T_BROWSER_TYPE) -> CookieJar:
    match browser:
        case "chrome":
            return browser_cookie3.chrome()
        case "firefox":
            return browser_cookie3.firefox()
        case "opera":
            return browser_cookie3.opera()
        case "edge":
            return browser_cookie3.edge()
        case "brave":
            return browser_cookie3.brave()
        case "chromium":
            return browser_cookie3.chromium()
        case "vivaldi":
            return browser_cookie3.vivaldi()
        case "safari":
            return browser_cookie3.safari()


def extract_course_name(url: str) -> None | Tuple[str, str]:
    result = urlparse(url)
    netloc_split = result.netloc.split(".")
    if len(netloc_split) <= 2:
        return None
    subdomain = netloc_split[0]
    path_split = result.path.split("/")
    if len(path_split) <= 2:
        return None
    course_name = path_split[2]
    return subdomain, course_name


class UdemyClient:
    session: Session
    logger: Logger

    def __init__(
        self, bearer_token: str | None, browser: T_BROWSER_TYPE | None, logger: Logger
    ):
        self.logger = logger
        if bearer_token:
            session = Session(cookieJar=None)
            session._set_auth_headers(bearer_token=bearer_token)
            self.session = session
        else:
            if browser is None:
                logger.error(
                    "No bearer token was provided, and no browser for cookie extraction was specified."
                )
                sys.exit(1)
            logger.warning(
                "No bearer token was provided, attempting to use browser cookies."
            )
            self.session = Session(cookieJar=browser_to_cookie(browser))

    def _extract_subscription_course_info(self, url: str) -> int:
        response = self.session._get(url, self.logger)
        if response is None:
            self.logger.fatal("Unable to get a response")
            sys.exit(1)
        course_html = response.text
        soup = BeautifulSoup(course_html, "lxml")
        data = soup.find("div", {"class": "ud-component--course-taking--app"})
        match data:
            case Tag():
                data_args = data.attrs["data-module-args"]
                data_json = json.loads(data_args)
                course_id = data_json.get("courseId", None)
                return course_id
            case NavigableString():
                self.logger.fatal(
                    "Extracted a NavigableString. Expected a Tag instead. Exiting..."
                )
            case None:
                self.logger.fatal(
                    "Unable to extract arguments from course page! Make sure you have a cookies.txt file!"
                )
        self.session.terminate()
        sys.exit(1)

    def _extract_course_info_json(
        self, subdomain: str, url: str, course_id: int
    ) -> extract_course_info_json.Model:
        self.session._headers.update({"Referer": url})
        url = constants.COURSE_INFO_URL.format(subdomain=subdomain, course_id=course_id)
        try:
            response = self.session._get(url, self.logger)
            if response is None:
                self.logger.fatal("Unable to get a response")
                sys.exit(1)
            # can throw
            resp = extract_course_info_json.Model.model_validate_json(response.content)
        except ConnectionError as error:
            self.logger.fatal(f"Udemy Says: Connection error, {error}")
            sys.exit(1)
        return resp

    def _extract_large_course_assets_content(
        self, api_url: str
    ) -> extract_course_json.Model:
        api_url = (
            api_url.replace("10000", "50") if api_url.endswith("10000") else api_url
        )
        try:
            response = self.session._get(api_url, self.logger)
            if response is None:
                self.logger.fatal("Unable to get a response")
                sys.exit(1)
            base_data = extract_course_json.Model.model_validate_json(response.content)
        except ConnectionError as error:
            self.logger.fatal(f"Udemy Says: Connection error, {error}")
            sys.exit(1)
        else:
            _next = base_data.next
            while _next:
                self.logger.info("> Downloading course information.. ")
                try:
                    response = self.session._get(_next, self.logger)
                    if response is None:
                        self.logger.fatal("Unable to get a response")
                        sys.exit(1)
                    data = extract_course_json.Model.model_validate_json(
                        response.content
                    )
                except ConnectionError as error:
                    self.logger.fatal(f"Udemy Says: Connection error, {error}")
                    sys.exit(1)
                else:
                    _next = data.next
                    results = data.results
                    for d in results:
                        base_data.results.append(d)
            return base_data

    def _extract_course_assets_json(
        self, course_url: str, course_id: int, subdomain: str
    ):
        self.session._headers.update({"Referer": course_url})
        api_url = constants.COURSE_URL.format(subdomain=subdomain, course_id=course_id)
        try:
            resp = self.session._get_allow_5xx(api_url, self.logger)
            if resp is None:
                self.logger.fatal("Unable to get a response")
                sys.exit(1)
            if resp.status_code in [502, 503]:
                self.logger.info(
                    "> The course content is large, using large content extractor..."
                )
                resp = self._extract_large_course_assets_content(api_url=api_url)
            else:
                return extract_course_json.Model.model_validate_json(resp.content)
        except ConnectionError as error:
            self.logger.fatal(f"Udemy Says: Connection error, {error}")
            sys.exit(1)
        except (ValueError, Exception):
            return self._extract_large_course_assets_content(api_url=api_url)
        else:
            return resp

    def _extract_course_info(
        self, course_url: str
    ) -> Tuple[extract_course_info_json.Model, str]:
        logger = self.logger
        extract_result = extract_course_name(course_url)
        if extract_result is None:
            logger.warn("course_url is invalid:", course_url)
            sys.exit(1)
        subdomain, course_name = extract_result
        course: extract_course_info_json.Model | None = None
        course_id = self._extract_subscription_course_info(course_url)
        course = self._extract_course_info_json(subdomain, course_url, course_id)
        if course:
            return course, subdomain
        logger.fatal("Downloading course information, course id not found .. ")
        logger.fatal(
            "It seems either you are not enrolled or you have to visit the course atleast once while you are logged in.",
        )
        logger.info(
            "Terminating Session...",
        )
        self.session.terminate()
        logger.info(
            "Session terminated.",
        )
        sys.exit(1)


def check_for_ffmpeg(logger: Logger):
    try:
        subprocess.Popen(
            ["ffmpeg", "-version"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL
        ).wait()
        return True
    except FileNotFoundError:
        return False
    except Exception:
        logger.exception(
            "> Unexpected exception while checking for FFmpeg, please tell the program author about this!"
        )
        return True


class ChapterMetadata(BaseModel):
    title: str
    object_index: int
    id: int


def result_to_chapter_metadata(result: filtered_chapters.ModelItem) -> ChapterMetadata:
    return ChapterMetadata(
        id=result.id, title=result.title, object_index=result.object_index
    )


class Chapter(BaseModel):
    meta: ChapterMetadata | None
    assets: List[
        filtered_lectures.ModelItem
        | filtered_quizzes.ModelItem
        | filtered_practices.ModelItem
    ]


class Course(BaseModel):
    info: extract_course_info_json.Model
    chapters: List[Chapter]
    subdomain: str

    def total_lectures(self) -> int:
        return functools.reduce(
            operator.add, map(lambda ch: len(ch.assets), self.chapters)
        )

    def total_chapters(self) -> int:
        return len(self.chapters)


def transform_assets_list_to_chapters_list(
    results: List[extract_course_json.Result],
) -> List[Chapter]:
    chapters: List[Chapter] = []
    if not results:
        return chapters
    if results[0].field_class != "chapter":
        cur_chapter: Chapter = Chapter(meta=None, assets=[])
        first_non_chapter = 0
    else:
        cur_chapter: Chapter = Chapter(
            meta=result_to_chapter_metadata(results[0]), assets=[]
        )
        first_non_chapter = 1
    for entry in results[first_non_chapter:]:
        if entry.field_class == "chapter":
            chapters.append(cur_chapter)
            cur_chapter = Chapter(meta=result_to_chapter_metadata(entry), assets=[])
        else:
            cur_chapter.assets.append(entry)
    chapters.append(cur_chapter)
    return chapters


def get_chapter_folder_name(chapter: Chapter, course: Course) -> Tuple[int, str]:
    meta = chapter.meta
    if meta:
        chapter_index = meta.object_index
        chapter_folder_name = "{0:02}_".format(chapter_index) + slugify(
            sanitize_filename(meta.title)
        )
    else:
        # dummy chapter to handle lectures without chapters
        chapter_index = 0
        chapter_folder_name = "{0:02d}_".format(chapter_index) + slugify(
            sanitize_filename(course.info.published_title)
        )
    return chapter_index, chapter_folder_name


class QuizDL(BaseModel):
    file_name: str
    id: int
    url: str
    type: Literal["practice-test", "coding-exercise", "simple-quiz"]


class MasterPlaylistDL(BaseModel):
    base_name: str
    file_name: str  # should go in playlist_course_folder instead of course_folder
    url: str


class IndexPlaylistDL(BaseModel):
    base_name: str
    file_name: str  # should go in playlist_course_folder
    master_file_name: str  # parse this file to extract url, then download


class VideoEncryptedDL(BaseModel):
    index_file_name: str  # parse file -> extract segments -> download the segments
    file_name: str  # ends in .enc.mp4
    base_name: str


class VideoMp4DL(BaseModel):
    url: str
    file_name: str
    base_name: str


class VideoRegularDL(BaseModel):
    index_file_name: str  # parse file -> extract segments -> download the segments
    file_name: str  # ends in .ts
    base_name: str


class VideoDecryptEmbedSubs(BaseModel):
    encrypted_file_name: str  # ends in .enc.mp4
    decrypted_file_name: str  # ends in .mp4
    file_name: str  # ends in .sub.mp4
    base_name: str


class VideoDecrypt(BaseModel):
    encrypted_file_name: str  # ends in .mp4
    file_name: str  # ends in .mp4
    base_name: str


class EmbedSubs(BaseModel):
    file_name: str  # ends in .sub.mp4
    base_name: str
    preembed_file_name: str  # ends in .ts


class EmbedMp4Subs(BaseModel):
    file_name: str  # ends in .sub.mp4
    base_name: str
    preembed_file_name: str  # ends in .mp4


class ArticleWrite(BaseModel):
    file_name: str
    contents: str


class CaptionDL(BaseModel):
    file_name: str
    url: str


class FileDL(BaseModel):
    file_name: str
    url: str


class ExternalLinkWrite(BaseModel):
    file_name: str
    contents: str


AssetTask = (
    ArticleWrite
    | QuizDL
    | CaptionDL
    | FileDL
    | ExternalLinkWrite
    | MasterPlaylistDL
    | IndexPlaylistDL
    | VideoDecrypt
    | VideoRegularDL
    | VideoEncryptedDL
    | VideoDecrypt
    | VideoDecryptEmbedSubs
    | VideoMp4DL
    | EmbedSubs
    | EmbedMp4Subs
)


def asset_to_comparable(asset: AssetTask) -> int:
    match asset:
        case CaptionDL():
            return 1
        case FileDL():
            return 2
        case MasterPlaylistDL():
            return 3
        case IndexPlaylistDL():
            # must come after master playlist
            return 4
        case ExternalLinkWrite():
            return 5
        case ArticleWrite():
            return 6
        case VideoMp4DL():
            return 7
        case VideoRegularDL():
            # must come after caption and index playlist
            return 8
        case VideoEncryptedDL():
            # must come after caption and index playlist
            return 9
        case VideoDecrypt():
            # must come after VideoEncryptedDL
            return 10
        case VideoDecryptEmbedSubs():
            # must come after VideoEncryptedDL
            return 11
        case EmbedSubs():
            # must come after VideoDecrypt, VideoNormalDL. should be independent from VideoDecryptEmbedSubs
            return 12
        case EmbedMp4Subs():
            # must come after VideoMp4DL()
            return 13
        case QuizDL():
            return 14


class LectureGroupDL(BaseModel):
    prefix_id: int
    lecture_index: int
    results: List[AssetTask] = []


def find_hls_stream(
    sources: List[filtered_lectures.MediaSource],
) -> filtered_lectures.MediaSource | None:
    for url in sources:
        if url.type == "application/x-mpegURL":
            return url
    return None


def find_best_mp4(
    sources: List[filtered_lectures.MediaSource], encrypted: bool
) -> filtered_lectures.MediaSource | None:
    if encrypted:
        return None
    if not sources:
        return None
    urls = sorted(sources, key=lambda url: int(url.label.split("x")[0]))
    best_url = urls[-1]
    if best_url.type == "video/mp4":
        return best_url
    return None


def create_lecture_group_dl(
    prefix_id: int,
    lecture_index: int,
    entry: (
        filtered_lectures.ModelItem
        | filtered_quizzes.ModelItem
        | filtered_practices.ModelItem
    ),
    state: State,
    subdomain: str,
) -> LectureGroupDL:
    asset_list: List[AssetTask] = []
    logger = state.logger
    match entry:
        case filtered_lectures.ModelItem():
            main_asset = entry.asset
            if main_asset.asset_type == "Article":
                pre_base = slugify(sanitize_filename(entry.title))
                article_file_name = f"{prefix_id:03}_{pre_base}.html"
                result = ArticleWrite(
                    contents=main_asset.body, file_name=article_file_name
                )
                asset_list.append(result)
            elif main_asset.asset_type == "Video":
                pre_base = slugify(sanitize_filename(entry.title))
                base_name = f"{prefix_id:03}_{pre_base}"
                master_m3u8_name = f"{base_name}.master.m3u8"
                index_m3u8_name = f"{base_name}.index.m3u8"
                encrypted = main_asset.stream_urls is None
                urls = main_asset.media_sources
                hls_stream = find_hls_stream(urls)
                encrypted_name = get_encrypted_video_name(base_name)
                embed_subs_name = get_embedsubs_video_name(base_name)
                if not hls_stream:
                    best_url = find_best_mp4(urls, encrypted)
                    if best_url:
                        mp4_name = f"{base_name}.mp4"
                        link = best_url.src
                        mp4_dl = VideoMp4DL(
                            url=link, base_name=base_name, file_name=mp4_name
                        )
                        asset_list.append(mp4_dl)
                        if state.embed_subs:
                            mp4_embed = EmbedMp4Subs(
                                file_name=embed_subs_name,
                                base_name=base_name,
                                preembed_file_name=mp4_name,
                            )
                            asset_list.append(mp4_embed)
                    else:
                        logger.warning(
                            f"unable to find hls or mp4 stream for {main_asset.title}"
                        )
                else:
                    final_url = hls_stream.src
                    master_dl = MasterPlaylistDL(
                        file_name=master_m3u8_name, url=final_url, base_name=base_name
                    )
                    index_dl = IndexPlaylistDL(
                        file_name=index_m3u8_name,
                        master_file_name=master_m3u8_name,
                        base_name=base_name,
                    )
                    asset_list.extend([master_dl, index_dl])
                    if encrypted:
                        decrypted_name = get_decrypted_video_name(base_name)
                        vid_dl = VideoEncryptedDL(
                            file_name=encrypted_name,
                            index_file_name=index_m3u8_name,
                            base_name=base_name,
                        )
                        asset_list.append(vid_dl)
                        if state.keys:
                            if state.embed_subs:
                                decrypt_task = VideoDecryptEmbedSubs(
                                    base_name=base_name,
                                    encrypted_file_name=encrypted_name,
                                    decrypted_file_name=decrypted_name,
                                    file_name=embed_subs_name,
                                )
                            else:
                                decrypt_task = VideoDecrypt(
                                    base_name=base_name,
                                    encrypted_file_name=encrypted_name,
                                    file_name=decrypted_name,
                                )
                            asset_list.append(decrypt_task)
                    else:
                        regulardl_name = get_regulardl_video_name(base_name)
                        vid_dl = VideoRegularDL(
                            file_name=regulardl_name,
                            index_file_name=index_m3u8_name,
                            base_name=base_name,
                        )
                        asset_list.append(vid_dl)
                        if state.embed_subs:
                            embed_subs_task = EmbedSubs(
                                base_name=base_name,
                                file_name=embed_subs_name,
                                preembed_file_name=regulardl_name,
                            )
                            asset_list.append(embed_subs_task)
                for caption in entry.asset.captions:
                    cap_lang = slugify(caption.locale_id.split("_")[0])
                    if state.caption_locale in (cap_lang, "all"):
                        cap_file_name = f"{base_name}_{cap_lang}.vtt"
                        cap_dl = CaptionDL(file_name=cap_file_name, url=caption.url)
                        asset_list.append(cap_dl)
            else:
                logger.warning(
                    f"unknown lecture asset type {main_asset.asset_type} for {main_asset.title}"
                )
            for sup_asset in entry.supplementary_assets:
                match sup_asset.asset_type:
                    case "File":
                        if sup_asset.download_urls:
                            file = sup_asset.download_urls.File[0]
                            if file:
                                url = file.file
                                filename = sup_asset.filename
                                filedl_name = f"{prefix_id:03}_{filename}"
                                file_dl = FileDL(file_name=filedl_name, url=url)
                                asset_list.append(file_dl)
                            else:
                                logger.warning(
                                    f"file has no download links: {sup_asset.title}"
                                )
                    case "ExternalLink":
                        filename = sup_asset.filename
                        link_file_name = f"{prefix_id:03}_{filename}.url"
                        url = sup_asset.external_url
                        external_link = ExternalLinkWrite(
                            file_name=link_file_name, contents=url
                        )
                        asset_list.append(external_link)
                    case other:
                        logger.warning(
                            f"unknown supplementary asset type {other} for {sup_asset.title}"
                        )
        case filtered_quizzes.ModelItem():
            pre_base = slugify(sanitize_filename(entry.title))
            article_file_name = f"{prefix_id:03}_{pre_base}.json"
            url = constants.QUIZ_URL.format(subdomain=subdomain, quiz_id=entry.id)
            quiz_asset: QuizDL | None = None
            if entry.type == "practice-test":
                quiz_asset = QuizDL(
                    file_name=article_file_name,
                    id=entry.id,
                    type="practice-test",
                    url=url,
                )
            elif entry.type == "coding-exercise":
                quiz_asset = QuizDL(
                    file_name=article_file_name,
                    id=entry.id,
                    type="coding-exercise",
                    url=url,
                )
            elif entry.type == "simple-quiz":
                quiz_asset = QuizDL(
                    file_name=article_file_name,
                    id=entry.id,
                    type="simple-quiz",
                    url=url,
                )
            else:
                logger.warning(f"unknown quiz type {entry.type} for {entry.title}")
            if quiz_asset:
                asset_list.append(quiz_asset)
        case filtered_practices.ModelItem():
            logger.warning(
                f"> I haven't gotten around to implementing {entry.field_class}: {entry.title} (id: {entry.id})"
            )
        case default:
            assert_never(default)
    asset_list.sort(key=asset_to_comparable)
    return LectureGroupDL(
        prefix_id=prefix_id, lecture_index=lecture_index, results=asset_list
    )


class ChapterDL(BaseModel):
    chapter_folder_name: str
    chapter_idx: int  # 0 if dummy chapter, 1-indexed otherwise
    lecture_dl: List[LectureGroupDL]


class CourseDL(BaseModel):
    total_lectures: int
    total_chapters: int
    instructor: str
    course_folder_name: str
    chapter_dl: List[ChapterDL]


def transform_course_to_download_plan(
    course: Course, state: State, subdomain: str
) -> CourseDL:
    course_folder_name = (
        str(course.info.id) if state.id_as_course_name else course.info.published_title
    )
    lecture_counter = 0
    total_lectures = course.total_lectures()
    total_chapters = course.total_chapters()
    chapter_dl_list: List[ChapterDL] = []
    instructor = course.info.visible_instructors[0]
    if not instructor:
        state.logger.error(f"No instructors found for {course.info.url}")
        sys.exit(1)
    instructor_name = instructor.url.strip().strip("/").split("/")[-1]
    for chapter in course.chapters:
        if not state.use_continuous_lecture_numbers:
            lecture_counter = 0
        chapter_index, chapter_folder_name = get_chapter_folder_name(chapter, course)
        lecture_dl_list: List[LectureGroupDL] = []
        for entry in chapter.assets:
            lecture_counter += 1
            lecture_index = (
                -1
                if isinstance(entry, filtered_practices.ModelItem)
                else entry.object_index
            )
            lecture_group = create_lecture_group_dl(
                lecture_index=lecture_index,
                prefix_id=lecture_counter,
                entry=entry,
                state=state,
                subdomain=subdomain,
            )
            lecture_dl_list.append(lecture_group)
        chapter_dl = ChapterDL(
            chapter_folder_name=chapter_folder_name,
            chapter_idx=chapter_index,
            lecture_dl=lecture_dl_list,
        )
        chapter_dl_list.append(chapter_dl)
    return CourseDL(
        instructor=instructor_name,
        chapter_dl=chapter_dl_list,
        course_folder_name=course_folder_name,
        total_chapters=total_chapters,
        total_lectures=total_lectures,
    )


def download_video(
    asset: VideoRegularDL | VideoEncryptedDL,
    chapter_path: Path,
    playlist_chapter_path: Path,
    state: State,
):
    tmp_filepath = Path(chapter_path, f"{asset.file_name}.part").resolve()
    filepath = Path(chapter_path, asset.file_name)
    index_path = Path(playlist_chapter_path, asset.index_file_name)
    file_base = asset.base_name
    if not index_path.exists():
        state.logger.warning(f"index playlist {index_path} missing")
        return
    index_text = index_path.read_text()
    parsed = m3u8.loads(index_text)
    uris: List[str] = parsed.segments.uri
    tasks: List[Task] = []
    for i, uri in enumerate(uris):
        task = Task(path=Path(chapter_path, f"{file_base}_{i}.frag.ts"), url=uri)
        tasks.append(task)
    segment_dl = SegmentDL(path=tmp_filepath, tasks=tasks)
    download_segment_stream(segment_dl, state)
    if tmp_filepath.exists():
        tmp_filepath.rename(filepath)
    else:
        state.logger.warning(f"{tmp_filepath} not found. rename failed")


def get_asset_filepath(asset: AssetTask, chapter_path, playlist_chapter) -> Path:
    match asset:
        case (
            CaptionDL()
            | FileDL()
            | QuizDL()
            | ExternalLinkWrite()
            | ArticleWrite()
            | VideoEncryptedDL()
            | VideoRegularDL()
            | VideoDecrypt()
            | EmbedSubs()
            | VideoDecryptEmbedSubs()
            | VideoMp4DL()
            | EmbedMp4Subs()
        ):
            return Path(chapter_path, asset.file_name).resolve()
        case MasterPlaylistDL() | IndexPlaylistDL():
            return Path(playlist_chapter, asset.file_name).resolve()
        case default:
            assert_never(default)


def get_encrypted_video_name(base_name: str) -> str:
    return f"{base_name}.enc.ts"


def get_decrypted_video_name(base_name: str) -> str:
    return f"{base_name}.mp4"


def get_embedsubs_video_name(base_name: str) -> str:
    return f"{base_name}.sub.mp4"


def get_regulardl_video_name(base_name: str) -> str:
    return f"{base_name}.ts"


def get_encrypted_video_path(chapter_path: Path, base_name: str) -> Path:
    return Path(chapter_path, get_encrypted_video_name(base_name))


def get_decrypted_video_path(chaper_path: Path, base_name: str) -> Path:
    return Path(chaper_path, get_decrypted_video_name(base_name))


def get_regulardl_video_path(chapter_path: Path, base_name: str) -> Path:
    return Path(chapter_path, get_regulardl_video_name(base_name))


def get_embedsubs_video_path(chapter_path: Path, base_name: str) -> Path:
    return Path(chapter_path, get_embedsubs_video_name(base_name))


def should_skip_dl(asset: AssetTask, filepath: Path, chapter_path: Path) -> bool:
    match asset:
        case MasterPlaylistDL() | IndexPlaylistDL():
            base_name = asset.base_name
            embedsubs_path = get_embedsubs_video_path(chapter_path, base_name)
            # playlist files used by encrypted and regular encrypted files
            decrypted_path = get_decrypted_video_path(chapter_path, base_name)
            encrypted_path = get_encrypted_video_path(chapter_path, base_name)
            regulardl_path = get_regulardl_video_path(chapter_path, base_name)
            # playlist can become outdated, so we delete all of the course m3u8 files at the start
            return (
                filepath.exists()
                or embedsubs_path.exists()
                or decrypted_path.exists()
                or regulardl_path.exists()
                or encrypted_path.exists()
            )
        case VideoEncryptedDL():
            base_name = asset.base_name
            embedsubs_path = get_embedsubs_video_path(chapter_path, base_name)
            decrypted_path = get_decrypted_video_path(chapter_path, base_name)
            return (
                filepath.exists() or embedsubs_path.exists() or decrypted_path.exists()
            )
        case VideoRegularDL() | VideoDecrypt() | VideoMp4DL():
            base_name = asset.base_name
            embedsubs_path = get_embedsubs_video_path(chapter_path, base_name)
            return filepath.exists() or embedsubs_path.exists()
        case (
            CaptionDL()
            | FileDL()
            | QuizDL()
            | ExternalLinkWrite()
            | ArticleWrite()
            | EmbedSubs()
            | VideoDecryptEmbedSubs()
            | EmbedMp4Subs()
        ):
            return filepath.exists()
        case default:
            assert_never(default)


def create_decryption_file(key_pair: KeyIdPair, encrypted_file: Path) -> str:
    key = key_pair.key
    kid = key_pair.kid
    key_base64 = b64encode(bytes.fromhex(key)).decode()
    kid_upper = kid.upper()
    return f"""#EXTM3U
#EXT-X-VERSION:5
#EXT-X-TARGETDURATION:7
#EXT-X-PLAYLIST-TYPE:VOD
#EXT-X-KEY:METHOD=SAMPLE-AES,URI="data:application/octet-stream;base64,{key_base64}",IV=0x{kid_upper},KEYFORMAT="com.apple.streamingkeydelivery",KEYFORMATVERSIONS="1"
#EXTINF:0.1,
file://{str(encrypted_file)}
#EXT-X-ENDLIST
"""


def ffmpeg_decryption_args(tmp_filepath: Path) -> List[str]:
    # ffmpeg -y -loglevel info -allowed_extensions 'ALL' -protocol_whitelist 'crypto,data,pipe,file' -f hls -i pipe: -c:a copy -c:v copy -c:s mov_text -f mp4 hopeful.lmao
    return [
        "ffmpeg",
        "-y",
        "-loglevel",
        "warning",
        "-allowed_extensions",
        "ALL",
        "-protocol_whitelist",
        "crypto,data,pipe,file",
        "-f",
        "hls",
        "-i",
        "pipe:",
        "-c:a",
        "copy",
        "-c:v",
        "copy",
        "-c:s",
        "mov_text",
        "-f",
        "mp4",
        str(tmp_filepath),
    ]


def ffmpeg_decryption_embed_args(vtt_files: List[str], tmp_filepath: Path) -> List[str]:
    vtt_args: List[str] = []
    for vtt_file in vtt_files:
        vtt_args.extend(["-f", "webvtt", "-i", vtt_file])
    return [
        "ffmpeg",
        "-y",
        "-loglevel",
        "warning",
        "-allowed_extensions",
        "ALL",
        "-protocol_whitelist",
        "crypto,data,pipe,file",
        "-thread_queue_size",
        "8096",
        "-f",
        "hls",
        "-i",
        "pipe:",
        *vtt_args,
        "-c:a",
        "copy",
        "-c:v",
        "copy",
        "-c:s",
        "mov_text",
        "-f",
        "mp4",
        str(tmp_filepath),
    ]


def ffmpeg_embed_args(
    preembed_filepath: Path,
    vtt_files: List[str],
    tmp_filepath: Path,
    input_format: Literal["mpegts"] | Literal["mp4"],
) -> List[str]:
    # ffmpeg -y -loglevel info -f mp4 -i 001_collections1.mp4 -f webvtt -i 001_collections1_en.vtt -c:a copy -c:v copy -c:s mov_text -f mp4 embed.mp4
    vtt_args: List[str] = []
    for vtt_file in vtt_files:
        vtt_args.extend(["-f", "webvtt", "-i", vtt_file])
    return [
        "ffmpeg",
        "-y",
        "-loglevel",
        "warning",
        "-f",
        input_format,
        "-i",
        str(preembed_filepath),
        *vtt_args,
        "-c:a",
        "copy",
        "-c:v",
        "copy",
        "-c:s",
        "mov_text",
        "-f",
        "mp4",
        str(tmp_filepath),
    ]


def decrypt_video(
    asset: VideoDecrypt, filepath: Path, chapter_path: Path, state: State
):
    logger = state.logger
    encrypted_filepath = Path(chapter_path, asset.encrypted_file_name).resolve()
    if not encrypted_filepath.exists():
        logger.warning(
            f"encrypted file not found. skipping: {asset.encrypted_file_name}"
        )
        return
    keys = state.keys
    if not keys:
        logger.warning(
            f"no keys found skipping decryption: {asset.encrypted_file_name}"
        )
        return
    tmp_filepath = Path(chapter_path, f"{asset.file_name}.part").resolve()
    args = ffmpeg_decryption_args(tmp_filepath)
    decryption_file = create_decryption_file(keys, encrypted_filepath)
    process = subprocess.Popen(args, shell=False, stdin=subprocess.PIPE)
    process.communicate(input=str.encode(decryption_file))
    ret_code = process.wait()
    if ret_code != 0:
        logger.warning(
            f"ffmpeg ret_code nonzero {ret_code} for {asset.encrypted_file_name}. Are you using the correct key?"
        )
        tmp_filepath.unlink(missing_ok=True)
        return
    tmp_filepath.rename(filepath)
    encrypted_filepath.unlink(missing_ok=True)


def decrypt_video_embed_subs(
    asset: VideoDecryptEmbedSubs,
    filepath: Path,
    chapter_path: Path,
    encrypted_filepath: Path,
    state: State,
):
    logger = state.logger
    keys = state.keys
    if not keys:
        logger.warning(
            f"no keys found skipping decryption: {asset.encrypted_file_name}"
        )
        return
    tmp_filepath = Path(chapter_path, f"{asset.file_name}.part").resolve()
    search_pattern = str(Path(chapter_path, f"{asset.base_name}_*.vtt").resolve())
    vtt_files = glob(search_pattern)
    args = ffmpeg_decryption_embed_args(vtt_files, tmp_filepath)
    decryption_file = create_decryption_file(keys, encrypted_filepath)
    process = subprocess.Popen(args, shell=False, stdin=subprocess.PIPE)
    process.communicate(input=str.encode(decryption_file))
    ret_code = process.wait()
    if ret_code != 0:
        logger.warning(
            f"ffmpeg ret_code nonzero {ret_code} for {asset.encrypted_file_name}. Are you using the correct key?"
        )
        tmp_filepath.unlink(missing_ok=True)
        return
    tmp_filepath.rename(filepath)
    encrypted_filepath.unlink(missing_ok=True)


def run_embed_subs(
    asset: EmbedSubs | VideoDecryptEmbedSubs | EmbedMp4Subs,
    filepath: Path,
    chapter_path: Path,
    preembed_file_name: str,
    logger: Logger,
):
    preembed_filepath = Path(chapter_path, preembed_file_name).resolve()
    if not preembed_filepath.exists():
        logger.warning(f"preembed file not found. skipping: {preembed_file_name}")
        return
    tmp_filepath = Path(chapter_path, f"{asset.file_name}.part").resolve()
    search_pattern = str(Path(chapter_path, f"{asset.base_name}_*.vtt").resolve())
    vtt_files = glob(search_pattern)
    input_format = "mpegts" if isinstance(asset, EmbedSubs) else "mp4"
    args = ffmpeg_embed_args(preembed_filepath, vtt_files, tmp_filepath, input_format)
    process = subprocess.Popen(args, shell=False)
    ret_code = process.wait()
    if ret_code != 0:
        logger.warning(f"ffmpeg ret_code nonzero {ret_code} for {preembed_file_name}")
        tmp_filepath.unlink(missing_ok=True)
        return
    tmp_filepath.rename(filepath)
    preembed_filepath.unlink(missing_ok=True)


def download_asset(
    asset: AssetTask,
    state: State,
    session: Session,
    chapter_path: Path,
    playlist_chapter_path: Path,
):
    logger = state.logger
    filepath = get_asset_filepath(asset, chapter_path, playlist_chapter_path)
    if should_skip_dl(asset, filepath, chapter_path):
        logger.info(
            f"\t\tAsset or dependent asset resolved. Skipping: {type(asset).__name__:>17} -> {asset.file_name}"
        )
        return
    logger.info(f"\t\tResolving asset {type(asset).__name__:>17} -> {asset.file_name}")
    match asset:
        case CaptionDL() | FileDL() | MasterPlaylistDL():
            response = session._get(asset.url, logger)  # probably don't need cookies
            if not response:
                logger.warning(f"unable to fetch {asset.file_name}")
                return
            tmp_filepath = Path(chapter_path, f"{asset.file_name}.part").resolve()
            tmp_filepath.write_bytes(response.content)
            tmp_filepath.rename(filepath)
        case IndexPlaylistDL():
            master_path = Path(playlist_chapter_path, asset.master_file_name)
            if not master_path.exists():
                logger.warning(f"master playlist {asset.master_file_name} missing")
                return
            parsed = m3u8.loads(master_path.read_text())
            best_playlist: m3u8.Playlist = parsed.playlists[-1]
            response = session._get(
                best_playlist.uri, state.logger
            )  # probably don't need cookies from https://udemy.com behind CloudFlare
            if not response:
                state.logger.warning(
                    f"unable to fetch index playlist {asset.file_name} ({best_playlist.uri})"
                )
                return
            tmp_filepath = Path(chapter_path, f"{asset.file_name}.part").resolve()
            tmp_filepath.write_bytes(response.content)
            tmp_filepath.rename(filepath)
        case VideoDecrypt():
            decrypt_video(asset, filepath, chapter_path, state)
        case ExternalLinkWrite() | ArticleWrite():
            tmp_filepath = Path(chapter_path, f"{asset.file_name}.part").resolve()
            tmp_filepath.write_text(asset.contents)
            tmp_filepath.rename(filepath)
        case QuizDL():
            response = session._get(
                asset.url, logger
            )  # probably need cookies: from Udemy API
            if not response:
                logger.warning(f"unable to fetch {asset.file_name}")
                return
            tmp_filepath = Path(chapter_path, f"{asset.file_name}.part").resolve()
            tmp_filepath.write_bytes(response.content)
            tmp_filepath.rename(filepath)
        case VideoRegularDL() | VideoEncryptedDL():
            download_video(asset, chapter_path, playlist_chapter_path, state)
        case EmbedSubs():
            run_embed_subs(
                asset, filepath, chapter_path, asset.preembed_file_name, state.logger
            )
        case EmbedMp4Subs():
            run_embed_subs(
                asset, filepath, chapter_path, asset.preembed_file_name, state.logger
            )
        case VideoDecryptEmbedSubs():
            tmp_filepath = Path(chapter_path, f"{asset.file_name}.part").resolve()
            encrypted_filepath = Path(chapter_path, asset.encrypted_file_name)
            decrypted_filepath = Path(chapter_path, asset.decrypted_file_name)
            if encrypted_filepath.exists():
                # this is normal
                decrypt_video_embed_subs(
                    asset, filepath, chapter_path, encrypted_filepath, state
                )
            elif decrypted_filepath.exists():
                # happens iff ran with `--decrypt` and without `--embed` previously
                run_embed_subs(
                    asset,
                    filepath,
                    chapter_path,
                    asset.decrypted_file_name,
                    state.logger,
                )
            else:
                logger.warning(
                    f"files {decrypted_filepath} and {encrypted_filepath} missing"
                )
        case VideoMp4DL():
            tmp_filepath = Path(chapter_path, f"{asset.file_name}.part").resolve()
            download_mp4(asset.url, tmp_filepath, state)
            if tmp_filepath.exists():
                tmp_filepath.rename(filepath)
            else:
                state.logger.warning(f"{tmp_filepath} not found. rename failed")
        case _:
            assert_never(asset)


def prepare_dl_folders(dl_plan: CourseDL, state: State) -> Tuple[Path, Path]:
    playlist_dir = Path("temp").resolve()
    download_dir = state.download_dir
    relative_course_path = Path(dl_plan.instructor, dl_plan.course_folder_name)
    course_path = Path(download_dir, relative_course_path).resolve()
    playlist_course_path = Path(playlist_dir, relative_course_path).resolve()
    if playlist_course_path.exists():  # m3u8 files can become outdated
        for file in glob(str(playlist_course_path) + "/**/*.m3u8", recursive=True):
            Path(file).unlink()
    playlist_course_path.mkdir(parents=True, exist_ok=True)
    return course_path, playlist_course_path


def download(
    dl_plan: CourseDL,
    state: State,
    udemy_client: UdemyClient,
    course_path: Path,
    playlist_course_path: Path,
):
    logger = state.logger
    total_chapters = dl_plan.total_chapters
    total_lectures = dl_plan.total_lectures
    for chapter_dl in dl_plan.chapter_dl:
        logger.info(f"Downloading chapter {chapter_dl.chapter_idx} of {total_chapters}")
        chapter_path = Path(course_path, chapter_dl.chapter_folder_name)
        playlist_chapter_path = Path(
            playlist_course_path, chapter_dl.chapter_folder_name
        )
        playlist_chapter_path.mkdir(parents=True, exist_ok=True)
        chapter_path.mkdir(parents=True, exist_ok=True)
        for lesson in chapter_dl.lecture_dl:
            logger.info(
                f"\tDownloading lesson {lesson.lecture_index} of {total_lectures}"
            )
            for asset in lesson.results:
                download_asset(
                    asset,
                    state,
                    udemy_client.session,
                    chapter_path,
                    playlist_chapter_path,
                )


class AssetPaths(BaseModel):
    playlist_chapter_path: Path
    chapter_path: Path
    asset: AssetTask


def list_assets_with_paths(
    dl_plan: CourseDL, course_path: Path, playlist_course_path: Path
) -> List[AssetPaths]:
    result: List[AssetPaths] = []
    for chapter_dl in dl_plan.chapter_dl:
        chapter_path = Path(course_path, chapter_dl.chapter_folder_name)
        playlist_chapter_path = Path(
            playlist_course_path, chapter_dl.chapter_folder_name
        )
        playlist_chapter_path.mkdir(parents=True, exist_ok=True)
        chapter_path.mkdir(parents=True, exist_ok=True)
        for lesson in chapter_dl.lecture_dl:
            for asset in lesson.results:
                asset_paths = AssetPaths(
                    asset=asset,
                    chapter_path=chapter_path,
                    playlist_chapter_path=playlist_chapter_path,
                )
                result.append(asset_paths)
    return result


def tasks_for_master_playlists(
    list_assets: List[AssetPaths],
) -> List[Task]:
    results: List[Task] = []
    for entry in list_assets:
        asset = entry.asset
        if isinstance(asset, MasterPlaylistDL):
            filepath = get_asset_filepath(
                asset, entry.chapter_path, entry.playlist_chapter_path
            )
            if should_skip_dl(asset, filepath, entry.chapter_path):
                continue
            task = Task(
                path=filepath,
                url=asset.url,
            )
            results.append(task)
    return results


def tasks_for_index_playlist_assets(
    list_assets: List[AssetPaths], logger: Logger
) -> List[Task]:
    results: List[Task] = []
    for entry in list_assets:
        if isinstance(entry.asset, IndexPlaylistDL):
            asset = entry.asset
            filepath = get_asset_filepath(
                asset, entry.chapter_path, entry.playlist_chapter_path
            )
            if should_skip_dl(asset, filepath, entry.chapter_path):
                continue
            master_path = Path(entry.playlist_chapter_path, asset.master_file_name)
            if not master_path.exists():
                logger.warning(f"master playlist {asset.master_file_name} missing")
                continue
            parsed = m3u8.loads(master_path.read_text())
            best_playlist: m3u8.Playlist = parsed.playlists[-1]
            task = Task(
                path=filepath,
                url=best_playlist.uri,
            )
            results.append(task)
    return results


def run_program(state: State):
    logger = state.logger
    load_from_file = state.load_from_file
    save_to_file = state.save_to_file
    if not check_for_ffmpeg(logger):
        logger.fatal("> FFMPEG is missing from your system or path!")
        sys.exit(1)
    if load_from_file and save_to_file:
        logger.info(
            "> 'load_from_file' and 'save_to_file' selected, not valid. exiting"
        )
        sys.exit(1)
    if load_from_file:
        logger.info(
            "> 'load_from_file' was specified, data will be loaded from json files instead of fetched"
        )
    if save_to_file:
        logger.info("> 'save_to_file' was specified, data will be saved to json files")
    dotenv.load_dotenv()
    if not state.bearer_token:
        state.bearer_token = os.getenv("UDEMY_BEARER")
    udemy_client = UdemyClient(state.bearer_token, state.browser, logger)
    udemy_path = Path("saved", "udemy.json").resolve()
    if load_from_file:
        logger.info("> Loading course information from file")
        # getting course assets json
        all_course_info = Course.model_validate_json(udemy_path.read_bytes())
    else:
        logger.info(f"> Fetching course information for {state.course_url}")
        # fetching course info
        course_info, subdomain = udemy_client._extract_course_info(state.course_url)
        course_id = course_info.id
        logger.info(
            f"> Course information fetched: published title is '{course_info.published_title}' and course id is '{course_id}'"
        )
        logger.info("> Fetching list of assets, this may take a minute...")
        # getting course assets json
        course_assets_json = udemy_client._extract_course_assets_json(
            state.course_url, course_id, subdomain
        )
        logger.info(
            f"> Course assets retrieved: found {len(course_assets_json.results)} base assets"
        )
        chapters = transform_assets_list_to_chapters_list(course_assets_json.results)
        all_course_info = Course(
            chapters=chapters, info=course_info, subdomain=subdomain
        )
        if save_to_file:
            udemy_path.write_text(all_course_info.model_dump_json())
    if state.info:
        print(all_course_info.model_dump_json(indent=4))
    else:
        logger.info("> Creating download plan.")
        download_plan = transform_course_to_download_plan(
            all_course_info, state, all_course_info.subdomain
        )
        logger.info(
            f"> Downloading {download_plan.instructor}/{download_plan.course_folder_name} to {state.download_dir}"
        )
        course_path, playlist_course_path = prepare_dl_folders(download_plan, state)
        if state.batch:
            list_assets_paths = list_assets_with_paths(
                download_plan, course_path, playlist_course_path
            )
            independent_tasks = tasks_for_master_playlists(list_assets_paths)
            logger.info(f"> Downloading {len(independent_tasks)} master playlists")
            download_cloudflare_files(independent_tasks, state.concurrent_downloads)
            index_tasks = tasks_for_index_playlist_assets(
                list_assets_paths, state.logger
            )
            logger.info(f"> Downloading {len(index_tasks)} index playlists")
            download_cloudflare_files(index_tasks, state.concurrent_downloads)
        download(download_plan, state, udemy_client, course_path, playlist_course_path)


def cli_runner(args: Arguments) -> None:
    state = get_state(args)
    run_program(state)


def main():
    typed_argparse.Parser(Arguments).bind(cli_runner).run()


if __name__ == "__main__":
    main()
