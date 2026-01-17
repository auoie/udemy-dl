# README

## About

Downloads videos from Udemy, decrypts them, and embeds subtitles.
It's fast, asynchronously downloading small files and video segments.
DRM-protected videos require a Widevine key and key-id pair.

## Usage

### Requirements

Requires `python3` and `ffmpeg`. It uses `ffmpeg` for decryption and subtitle embedding.

### Local Install

```bash
python3 -m venv .venv
source ./.venv/bin/activate
python3 -m pip install poetry
python3 -m poetry install
```

```bash
$HOME/.pyenv/versions/3.12.11/bin/python3 -m venv .venv
source ./.venv/bin/activate
pip install poetry==1.8.5
```

### Keyfile

For decryption, a keyfile of the form

```json
{ "key": "kid" }
```

is required in `keyfile.json`.

### Example

```bash
udemy-dl --help
udemy-dl \
  --batch-playlists \
  --embed-subs \
  --decrypt \
  --concurrent-downloads 32 \
  --browser firefox \
  --course "$COURSE_URL"
```

### Help

```text
usage: udemy-dl [-h] -c COURSE_URL [-b BEARER_TOKEN] [-l LANG] [--concurrent-downloads CONCURRENT_DOWNLOADS] [--batch-playlists] [--embed-subs] [--decrypt] [--info] [--id-as-course-name] [--save-to-file] [--load-from-file] [--log-level {DEBUG,INFO,ERROR,WARNING,CRITICAL}] [--browser {chrome,firefox,opera,edge,brave,chromium,vivaldi,safari}] [-o OUT] [--use-continuous-lecture-numbers]

options:
  -h, --help            show this help message and exit
  -c COURSE_URL, --course-url COURSE_URL
                        The URL of the course to download
  -b BEARER_TOKEN, --bearer-token BEARER_TOKEN
                        The Bearer token to use
  -l LANG, --lang LANG  The language to download for captions, specify 'all' to download all captions [default: en]
  --concurrent-downloads CONCURRENT_DOWNLOADS
                        The number of maximum concurrent downloads for batch downloads and segments (HLS and DASH, must be a number 1-30) [default: 16]
  --batch-playlists     Batch download master playlists. Then batch download index playlists. Then sequentially download all other assets as normal.
  --embed-subs          If a video has any subs, embed those subs in the output video
  --decrypt             Decrypt the encrypted videos. Requires a {key: kid} pair in `keyfile.json`
  --info                If specified, only course information will be printed, nothing will be downloaded
  --id-as-course-name   If specified, the course id will be used in place of the course name for the output directory. This is a 'hack' to reduce the path length
  --save-to-file        If specified, course content will be saved to a file that can be loaded later with --load-from-file, this can reduce processing time (Note that asset links expire after a certain amount of time)
  --load-from-file      If specified, course content will be loaded from a previously saved file with --save-to-file, this can reduce processing time (Note that asset links expire after a certain amount of time)
  --log-level {DEBUG,INFO,ERROR,WARNING,CRITICAL}
                        Logging level: one of DEBUG, INFO, ERROR, WARNING, CRITICAL [default: INFO]
  --browser {chrome,firefox,opera,edge,brave,chromium,vivaldi,safari}
                        The browser to extract cookies from
  -o OUT, --out OUT     Set the path to the output directory [default: out_dir]
  --use-continuous-lecture-numbers
                        Use continuous lecture numbering instead of per-chapter
```

## Credits

- https://github.com/Puyodead1/udemy-downloader.

## Todo

- download the `"_class": "practice"` type course entry
- `ffmpeg` doesn't return an error status if the `key` and `kid` are incorrect. don't decrypt a video if the `key` and `kid` are incorrect.
- Handle e-books. Example: https://www.udemy.com/course/csharp-tutorial-for-beginners/learn/
