import asyncio
from pathlib import Path
import time
from typing import Coroutine, List

import aiohttp
from pydantic import BaseModel
import tqdm
from curl_cffi.requests import BrowserType, AsyncSession

from udemy_dl.state import State


# https://blog.jonlu.ca/posts/async-python-http
# https://isaacong.me/posts/tracking-progress-of-python-asyncio-coroutines/
async def gather_with_concurrency[T, R](n: int, tasks: List[Coroutine[T, T, R]]):
    semaphore = asyncio.Semaphore(n)

    async def sem_task(task: Coroutine[T, T, R]):
        async with semaphore:
            return await task

    async_tasks = [sem_task(task) for task in tasks]
    results: List[R] = []
    for future in tqdm.tqdm(asyncio.as_completed(async_tasks), total=len(async_tasks)):
        result = await future
        results.append(result)
    return results


class Task(BaseModel):
    path: Path
    url: str


class SegmentDL(BaseModel):
    tasks: List[Task]
    path: Path


async def download_cloudflare_files_async(tasks: List[Task], concurrency: int):
    impersonate = BrowserType.chrome
    session = AsyncSession(impersonate=impersonate)

    async def get_async(task: Task):
        for i in range(10):
            response = await session.get(task.url)
            response.content
            if response.ok:
                obj = response.content
                task.path.write_bytes(obj)
                response.close()
                return
            else:
                print(
                    f"failed {task.url}: {response.status_code} (retrying attempt {i})"
                )
                await asyncio.sleep(0.8)

    async_arr = [get_async(task) for task in tasks]
    await gather_with_concurrency(concurrency, async_arr)
    await session.close()


async def download_files_async(tasks: List[Task], concurrency: int):
    conn = aiohttp.TCPConnector(limit=4 * concurrency)
    session = aiohttp.ClientSession(connector=conn)

    async def get_async(task: Task):
        for i in range(10):
            async with session.get(task.url, ssl=True) as response:
                if response.ok:
                    obj = await response.read()
                    task.path.write_bytes(obj)
                    response.close()
                    return
                else:
                    print(
                        f"failed {task.url}: {response.status} (retrying attempt {i})"
                    )
                    await asyncio.sleep(0.8)

    async_arr = [get_async(task) for task in tasks]
    await gather_with_concurrency(concurrency, async_arr)
    await session.close()
    await conn.close()


def download_files(tasks: List[Task], concurrency: int):
    asyncio.run(download_files_async(tasks, concurrency))


def download_cloudflare_files(tasks: List[Task], concurrency: int):
    asyncio.run(download_cloudflare_files_async(tasks, concurrency))


def download_segment_stream(segment_dl: SegmentDL, state: State):
    filepath = segment_dl.path
    start = time.time()

    # download_files(segment_dl.tasks, state.concurrent_downloads)
    download_files(segment_dl.tasks, state.concurrent_downloads)
    segment_dl.path.unlink(missing_ok=True)
    with segment_dl.path.open("ab") as f:
        for task in segment_dl.tasks:
            f.write(task.path.read_bytes())
            task.path.unlink()

    end = time.time()
    duration = end - start
    mbs = ((filepath.stat().st_size) / 1024) / 1024
    MBpS = mbs / duration

    print(f"{mbs:.3g} MB in {duration:.3g}s: {MBpS:.3g} MB/s")
