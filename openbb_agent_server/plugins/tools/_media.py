"""Async media helpers shared by multimodal tool sources."""

from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("openbb_agent_server.tools._media")


class MediaError(RuntimeError):
    """Base class for media-pipeline failures."""


class MediaTooLargeError(MediaError):
    """Raise when a payload exceeds the caller's byte budget."""


class MediaFetchError(MediaError):
    """Raise when an HTTP fetch fails."""


class FFmpegUnavailableError(MediaError):
    """Raise when ffmpeg or ffprobe is not on PATH."""


@dataclass(frozen=True)
class FetchedMedia:
    """Hold raw bytes and content-type for a resolved clip.

    Attributes
    ----------
    data : bytes
        The full media payload as downloaded or decoded.
    mime : str
        The resolved MIME type (e.g. ``"audio/ogg"``), without any
        ``charset`` or other parameters.
    """

    data: bytes
    mime: str

    @property
    def size_bytes(self) -> int:
        """Return the payload size in bytes.

        Returns
        -------
        int
            The length of `data`.
        """
        return len(self.data)


async def fetch_url(
    url: str,
    *,
    max_bytes: int,
    timeout_s: float = 60.0,
    fallback_mime: str = "application/octet-stream",
) -> FetchedMedia:
    """Stream ``url`` to bytes with a hard size cap.

    Download the response body chunk by chunk, aborting as soon as the
    accumulated size would exceed ``max_bytes``. The MIME type is taken
    from the ``Content-Type`` response header when present, otherwise
    guessed from the URL extension, otherwise ``fallback_mime``.

    Parameters
    ----------
    url : str
        The HTTP(S) URL to fetch.
    max_bytes : int
        Maximum number of bytes to accept. The download is aborted with
        :class:`MediaTooLargeError` once this would be exceeded.
    timeout_s : float, default 60.0
        Per-request timeout in seconds passed to the HTTP client.
    fallback_mime : str, default "application/octet-stream"
        MIME type used when neither the response header nor the URL
        extension yields one.

    Returns
    -------
    FetchedMedia
        The downloaded bytes paired with the resolved MIME type.

    Raises
    ------
    MediaError
        If ``httpx`` is not installed.
    MediaTooLargeError
        If the response body exceeds ``max_bytes``.
    MediaFetchError
        If the HTTP request fails (non-2xx status, connection or stream
        error).
    """
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover
        raise MediaError(
            "media fetch requires httpx; install agent_server[http]"
        ) from exc

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        try:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                mime = _detect_mime(
                    response.headers.get("content-type"), url, fallback_mime
                )
                buf = bytearray()
                async for chunk in response.aiter_bytes():
                    if not chunk:  # pragma: no cover
                        continue
                    if len(buf) + len(chunk) > max_bytes:
                        raise MediaTooLargeError(
                            f"remote media at {url!r} exceeds the configured "
                            f"max of {max_bytes} bytes "
                            f"(already received {len(buf)})"
                        )
                    buf.extend(chunk)
        except (httpx.HTTPError, httpx.StreamError) as exc:
            raise MediaFetchError(f"failed to fetch {url!r}: {exc}") from exc

    return FetchedMedia(data=bytes(buf), mime=mime)


async def to_data_url(
    raw: bytes,
    *,
    mime: str,
    max_bytes: int | None = None,
) -> str:
    """Encode ``raw`` as a ``data:`` URL off the event loop.

    The base64 encoding runs in a worker thread so large payloads do not
    block the running event loop.

    Parameters
    ----------
    raw : bytes
        The media payload to encode.
    mime : str
        The MIME type to embed in the resulting ``data:`` URL.
    max_bytes : int or None, optional
        When set, reject payloads larger than this many bytes before
        encoding.

    Returns
    -------
    str
        A ``data:<mime>;base64,<encoded>`` URL string.

    Raises
    ------
    MediaTooLargeError
        If ``max_bytes`` is set and ``raw`` exceeds it.
    """
    if max_bytes is not None and len(raw) > max_bytes:
        raise MediaTooLargeError(
            f"media exceeds the configured max of {max_bytes} bytes (got {len(raw)})"
        )
    b64 = await asyncio.to_thread(_b64_encode_to_str, raw)
    return f"data:{mime};base64,{b64}"


def _b64_encode_to_str(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _detect_mime(header: str | None, url: str, fallback: str) -> str:
    if header:
        return header.split(";", 1)[0].strip()
    guessed, _ = mimetypes.guess_type(url)
    return guessed or fallback


def _have_binary(name: str) -> bool:
    return shutil.which(name) is not None


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FFmpegUnavailableError(
            f"this operation needs {name} on $PATH. Install ffmpeg "
            "(``brew install ffmpeg`` / ``apt-get install ffmpeg``)."
        )
    return path


async def probe_audio_duration(raw: bytes) -> float:
    """Return the clip's duration in seconds via ``ffprobe``.

    Write ``raw`` to a temporary file and invoke ``ffprobe`` to read the
    container's ``format=duration`` field. The temporary file is always
    removed before returning.

    Parameters
    ----------
    raw : bytes
        The encoded audio (or media) payload to probe.

    Returns
    -------
    float
        The clip duration in seconds.

    Raises
    ------
    FFmpegUnavailableError
        If ``ffprobe`` is not on ``$PATH``.
    MediaError
        If ``ffprobe`` exits non-zero or returns a non-numeric duration.
    """
    import contextlib
    import os
    import tempfile

    ffprobe = _require_binary("ffprobe")
    fd, path = tempfile.mkstemp(prefix="oas-probe-", suffix=".bin")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        proc = await asyncio.create_subprocess_exec(
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            "-i",
            path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)
    if proc.returncode != 0:
        raise MediaError(
            f"ffprobe failed (rc={proc.returncode}): "
            f"{stderr.decode('utf-8', 'replace').strip()}"
        )
    try:
        return float(stdout.decode("ascii", "replace").strip())
    except ValueError as exc:
        raise MediaError(f"ffprobe returned non-numeric duration: {stdout!r}") from exc


@dataclass(frozen=True)
class AudioSegment:
    """Hold one slice of a clip produced by :func:`split_audio_bytes`.

    Attributes
    ----------
    index : int
        Zero-based position of this segment within the split.
    total : int
        Total number of segments the clip was split into.
    start_s : float
        Start offset of this segment within the source clip, in seconds.
    end_s : float
        End offset of this segment within the source clip, in seconds.
    data : bytes
        The encoded bytes of this segment.
    mime : str
        The MIME type of the segment (``"audio/<container>"``).
    """

    index: int
    total: int
    start_s: float
    end_s: float
    data: bytes
    mime: str


async def split_audio_bytes(
    raw: bytes,
    *,
    max_seconds: float,
    target_codec: str = "libopus",
    target_container: str = "ogg",
) -> list[AudioSegment]:
    """Split ``raw`` into segments no longer than ``max_seconds`` each.

    Probe the clip duration first; if it already fits within
    ``max_seconds`` the original bytes are returned as a single segment
    untouched. Otherwise the clip is cut into evenly sized windows and
    each window is re-encoded with ``ffmpeg`` to the requested codec and
    container (downmixed to mono at 24 kHz, 32 kbit/s).

    Parameters
    ----------
    raw : bytes
        The encoded audio payload to split.
    max_seconds : float
        Maximum duration of each segment in seconds. Must be greater
        than zero.
    target_codec : str, default "libopus"
        The ``ffmpeg`` audio codec to encode each cut segment with.
    target_container : str, default "ogg"
        The ``ffmpeg`` output container format for each segment; also
        used to build the segment MIME type.

    Returns
    -------
    list of AudioSegment
        One segment per window, in order. A clip shorter than
        ``max_seconds`` yields a single segment wrapping the original
        bytes.

    Raises
    ------
    ValueError
        If ``max_seconds`` is not greater than zero.
    FFmpegUnavailableError
        If ``ffprobe`` or ``ffmpeg`` is not on ``$PATH``.
    MediaError
        If probing or any ``ffmpeg`` re-encode fails.
    """
    if max_seconds <= 0:
        raise ValueError("max_seconds must be > 0")
    duration = await probe_audio_duration(raw)
    if duration <= max_seconds:
        return [
            AudioSegment(
                index=0,
                total=1,
                start_s=0.0,
                end_s=duration,
                data=raw,
                mime=f"audio/{target_container}",
            )
        ]
    n = int((duration + max_seconds - 1) // max_seconds)
    cuts: list[tuple[float, float]] = [
        (i * max_seconds, min((i + 1) * max_seconds, duration)) for i in range(n)
    ]
    return await _ffmpeg_split(
        raw,
        cuts=cuts,
        codec=target_codec,
        container=target_container,
    )


async def _ffmpeg_split(
    raw: bytes,
    *,
    cuts: Iterable[tuple[float, float]],
    codec: str,
    container: str,
) -> list[AudioSegment]:
    ffmpeg = _require_binary("ffmpeg")
    cuts_list = list(cuts)
    out: list[AudioSegment] = []
    for i, (start, end) in enumerate(cuts_list):
        proc = await asyncio.create_subprocess_exec(
            ffmpeg,
            "-loglevel",
            "error",
            "-y",
            "-i",
            "pipe:0",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-ac",
            "1",
            "-ar",
            "24000",
            "-c:a",
            codec,
            "-b:a",
            "32k",
            "-f",
            container,
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(input=raw)
        if proc.returncode != 0:
            raise MediaError(
                f"ffmpeg failed on segment {i} ({start:.1f}-{end:.1f}s): "
                f"{stderr.decode('utf-8', 'replace').strip()}"
            )
        out.append(
            AudioSegment(
                index=i,
                total=len(cuts_list),
                start_s=start,
                end_s=end,
                data=stdout,
                mime=f"audio/{container}",
            )
        )
    return out


def ffmpeg_available() -> bool:
    """Return ``True`` iff both ``ffmpeg`` and ``ffprobe`` are on ``$PATH``.

    Returns
    -------
    bool
        ``True`` when both binaries are resolvable, ``False`` if either
        is missing.
    """
    return _have_binary("ffmpeg") and _have_binary("ffprobe")


def flatten_message_content(content: Any) -> str:
    """Collapse a LangChain message ``content`` field to a single string.

    LangChain message content may be a plain string or a list of content
    blocks. For a list, only the textual parts are kept: ``str`` blocks
    are concatenated verbatim and ``dict`` blocks of ``type == "text"``
    contribute their ``"text"`` value; all other block types are
    ignored. Any other content value is coerced via `str`.

    Parameters
    ----------
    content : Any
        A LangChain message ``content`` value: a string, a list of
        string and/or content-block dicts, or any other object.

    Returns
    -------
    str
        The concatenated text, or an empty string when ``content`` is
        falsy.
    """
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content) if content else ""


__all__ = [
    "AudioSegment",
    "FFmpegUnavailableError",
    "FetchedMedia",
    "MediaError",
    "MediaFetchError",
    "MediaTooLargeError",
    "fetch_url",
    "ffmpeg_available",
    "flatten_message_content",
    "probe_audio_duration",
    "split_audio_bytes",
    "to_data_url",
]
