"""
Async API request processor with rate limiting and retries.
Inlined from agents.process_api_requests_from_file for self-contained medal package.
"""
import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import aiohttp
import tiktoken

from medal.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class StatusTracker:
    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0
    num_other_errors: int = 0
    time_of_last_rate_limit_error: float = 0.0


@dataclass
class APIRequest:
    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: object
    result: list = field(default_factory=list)

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ) -> None:
        logger.debug("Starting request #%s", self.task_id)
        error = None
        response = None
        try:
            async with session.post(
                url=request_url,
                headers=request_header,
                json=self.request_json["body"],
            ) as resp:
                response = await resp.json()
            if "error" in response:
                logger.warning("Request %s failed: %s", self.task_id, response.get("error"))
                status_tracker.num_api_errors += 1
                error = response
                if "rate limit" in str(response.get("error", {}).get("message", "")).lower():
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1
        except Exception as e:
            logger.warning("Request %s exception: %s", self.task_id, e)
            status_tracker.num_other_errors += 1
            error = e
        if error is not None:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logger.error("Request %s failed after all attempts", self.task_id)
                data = [self.task_id, self.request_json, [str(e) for e in self.result]]
                if self.metadata is not None:
                    data.append(self.metadata)
                _append_to_jsonl(data, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = [self.task_id, self.request_json, response]
            if self.metadata is not None:
                data.append(self.metadata)
            _append_to_jsonl(data, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1


def _append_to_jsonl(data: list, filename: str) -> None:
    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def _num_tokens_consumed_from_request(
    request_json: dict,
    _api_endpoint: str,
    token_encoding_name: str,
) -> int:
    encoding = tiktoken.get_encoding(token_encoding_name)
    max_tokens = request_json.get("max_tokens", 15)
    n = request_json.get("n", 1)
    completion_tokens = n * max_tokens
    num_tokens = 0
    for message in request_json.get("body", {}).get("messages", []):
        num_tokens += 4
        for key, value in message.items():
            if isinstance(value, str):
                num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens -= 1
    num_tokens += 2
    return num_tokens + completion_tokens


def _task_id_generator():
    i = 0
    while True:
        yield i
        i += 1


async def process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str,
    request_url: str,
    api_key: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name: str,
    max_attempts: int,
    logging_level: int = logging.INFO,
) -> None:
    """Process API requests from a JSONL file with rate limiting and retries."""
    pause_after_rate_limit = 15
    sleep_each_loop = 0.001
    request_header = {"Authorization": f"Bearer {api_key}"}
    if "/deployments" in request_url:
        request_header = {"api-key": api_key}
    queue: asyncio.Queue = asyncio.Queue()
    task_id_gen = _task_id_generator()
    status = StatusTracker()
    next_request = None
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()
    file_not_finished = True
    with open(requests_filepath, encoding="utf-8") as f:
        request_lines = f
        async with aiohttp.ClientSession() as session:
            while True:
                if next_request is None:
                    try:
                        next_request = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    if next_request is None and file_not_finished:
                        try:
                            line = next(request_lines)
                            request_json = json.loads(line)
                            tid = next(task_id_gen)
                            next_request = APIRequest(
                                task_id=tid,
                                request_json=request_json,
                                token_consumption=_num_tokens_consumed_from_request(
                                    request_json, request_url, token_encoding_name
                                ),
                                attempts_left=max_attempts,
                                metadata=request_json.pop("metadata", None),
                            )
                            status.num_tasks_started += 1
                            status.num_tasks_in_progress += 1
                        except StopIteration:
                            file_not_finished = False
                current_time = time.time()
                elapsed = current_time - last_update_time
                available_request_capacity = min(
                    available_request_capacity + max_requests_per_minute * elapsed / 60.0,
                    max_requests_per_minute,
                )
                available_token_capacity = min(
                    available_token_capacity + max_tokens_per_minute * elapsed / 60.0,
                    max_tokens_per_minute,
                )
                last_update_time = current_time
                if next_request is not None:
                    need = next_request.token_consumption
                    if available_request_capacity >= 1 and available_token_capacity >= need:
                        available_request_capacity -= 1
                        available_token_capacity -= need
                        next_request.attempts_left -= 1
                        asyncio.create_task(
                            next_request.call_api(
                                session=session,
                                request_url=request_url,
                                request_header=request_header,
                                retry_queue=queue,
                                save_filepath=save_filepath,
                                status_tracker=status,
                            )
                        )
                        next_request = None
                if status.num_tasks_in_progress == 0:
                    break
                await asyncio.sleep(sleep_each_loop)
                if time.time() - status.time_of_last_rate_limit_error < pause_after_rate_limit:
                    remaining = pause_after_rate_limit - (
                        time.time() - status.time_of_last_rate_limit_error
                    )
                    await asyncio.sleep(remaining)
                    logger.warning("Pausing after rate limit")
    logger.info("Parallel processing complete. Results saved to %s", save_filepath)
    if status.num_tasks_failed > 0:
        logger.warning(
            "%s / %s requests failed",
            status.num_tasks_failed,
            status.num_tasks_started,
        )
    if status.num_rate_limit_errors > 0:
        logger.warning("%s rate limit errors", status.num_rate_limit_errors)
