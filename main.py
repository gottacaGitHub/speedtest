from __future__ import annotations

import argparse
import statistics
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TextIO
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

BYTES_IN_MIB = 1024 * 1024
BITS_PER_BYTE = 8
BITS_PER_MBIT = 1_000_000
USER_AGENT = "PySpeedTest/1.0"
DEFAULT_ATTEMPTS = 10
DEFAULT_TIMEOUT = 60.0


# --------------------------------------------------------------- domain ---

@dataclass(frozen=True, slots=True)
class Attempt:
    """Один завершённый замер: полное время и байты на проводе (application bytes)."""
    elapsed_s: float
    wire_bytes: int


@dataclass(slots=True)
class Stats:
    """Агрегат по всем замерам."""
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def total_wire_bytes(self) -> int:
        return sum(a.wire_bytes for a in self.attempts)

    @property
    def total_elapsed_s(self) -> float:
        return sum(a.elapsed_s for a in self.attempts)

    @property
    def mean_elapsed_s(self) -> float:
        return statistics.fmean(a.elapsed_s for a in self.attempts)

    @property
    def median_elapsed_s(self) -> float:
        return statistics.median(a.elapsed_s for a in self.attempts)

    @property
    def stdev_elapsed_s(self) -> float:
        return statistics.stdev(a.elapsed_s for a in self.attempts) if len(self.attempts) > 1 else 0.0

    @property
    def throughput_bytes_per_s(self) -> float:
        return self.total_wire_bytes / self.total_elapsed_s if self.total_elapsed_s > 0 else 0.0


# ------------------------------------------------------------- transport ---

class Fetcher(Protocol):
    """Точка расширения: подменяется фейком в тестах без сети."""
    def fetch(self, url: str) -> Attempt: ...


ClientFactory = Callable[[], httpx.Client]


class HttpFetcher:
    """Один fetch = один GET с полным дочитыванием тела (application bytes)."""

    def __init__(
            self,
            client_factory: ClientFactory,
            *,
            close_after_fetch: bool,
            cache_bust: bool,
    ) -> None:
        self._factory = client_factory
        self._close_after = close_after_fetch
        self._cache_bust = cache_bust

    def fetch(self, url: str) -> Attempt:
        target = self._bust(url) if self._cache_bust else url
        headers = {"User-Agent": USER_AGENT}
        if self._cache_bust:
            headers["Cache-Control"] = "no-cache"
            headers["Pragma"] = "no-cache"

        client = self._factory()
        try:
            return self._download(client, target, headers)
        finally:
            if self._close_after:
                client.close()

    @staticmethod
    def _download(client: httpx.Client, url: str, headers: dict[str, str]) -> Attempt:
        wire = 0
        started = time.perf_counter()
        with client.stream("GET", url, headers=headers) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_raw():
                wire += len(chunk)
            elapsed = time.perf_counter() - started
        return Attempt(elapsed_s=elapsed, wire_bytes=wire)

    @staticmethod
    def _bust(url: str) -> str:
        parts = urlsplit(url)
        query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k != "nocache"]
        query.append(("nocache", uuid.uuid4().hex))
        return urlunsplit(parts._replace(query=urlencode(query)))


# --------------------------------------------------------- orchestration ---

class SpeedMeter:
    """Цикл N последовательных замеров."""

    def __init__(self, fetcher: Fetcher, *, attempts: int) -> None:
        if attempts < 1:
            raise ValueError("attempts must be >= 1")
        self._fetcher = fetcher
        self._attempts = attempts

    def run(self, url: str) -> Stats:
        stats = Stats()
        for _ in range(self._attempts):
            stats.attempts.append(self._fetcher.fetch(url))
        return stats


# --------------------------------------------------------------- reporter --

class Reporter(Protocol):
    def report(self, url: str, stats: Stats) -> None: ...


class ConsoleReporter:
    def __init__(self, out: TextIO | None = None) -> None:
        self._out = out if out is not None else sys.stdout

    def report(self, url: str, stats: Stats) -> None:
        w = self._out.write
        w(f"URL: {url}\n")
        w(f"Запросов: {len(stats.attempts)}\n")
        w("-\n")
        for i, a in enumerate(stats.attempts, start=1):
            w(f"  #{i:02d}  {a.elapsed_s:7.3f} s  {_fmt_bytes(a.wire_bytes)}\n")
        w("-\n")
        w(f"Суммарный объём:     {_fmt_bytes(stats.total_wire_bytes)}\n")
        w(f"Суммарное время:     {stats.total_elapsed_s:.3f} s\n")
        w(f"Среднее на запрос:   {stats.mean_elapsed_s:.3f} s\n")
        w(f"Медиана:             {stats.median_elapsed_s:.3f} s\n")
        w(f"Std. dev:            {stats.stdev_elapsed_s:.3f} s\n")
        speed = stats.throughput_bytes_per_s
        w(f"Скорость скачивания: {speed / BYTES_IN_MIB:.3f} MiB/s "
          f"({speed * BITS_PER_BYTE / BITS_PER_MBIT:.3f} Mbit/s)\n")


def _fmt_bytes(n: float) -> str:
    if n < 1024:
        return f"{int(n)} B"
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        n /= 1024
        if n < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}"
    return f"{n:.2f} TiB"


# -------------------------------------------------------------------- CLI ---

def _positive_int(s: str) -> int:
    v = int(s)
    if v < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return v


def _positive_float(s: str) -> float:
    v = float(s)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return v


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="speedtest.py",
        description="Скачивает URL N раз последовательно и печатает скорость скачивания.",
    )
    p.add_argument("url", help="адрес ресурса (например, тяжёлая картинка)")
    p.add_argument("-n", "--attempts", type=_positive_int, default=DEFAULT_ATTEMPTS,
                   help=f"число запросов (по умолчанию {DEFAULT_ATTEMPTS})")
    p.add_argument("--timeout", type=_positive_float, default=DEFAULT_TIMEOUT,
                   help=f"таймаут одного запроса, сек (по умолчанию {DEFAULT_TIMEOUT})")
    p.add_argument("--fresh-connection", action="store_true",
                   help="новое TCP/TLS-соединение на каждый запрос (по умолчанию keep-alive)")
    p.add_argument("--cache-bust", action=argparse.BooleanOptionalAction, default=True,
                   help="добавлять ?nocache=<uuid> и no-cache заголовки (по умолчанию вкл.)")
    return p


def _client_params(args: argparse.Namespace) -> dict[str, object]:
    return {"timeout": args.timeout, "follow_redirects": True, "http2": True}


def _build_fetcher(args: argparse.Namespace) -> tuple[HttpFetcher, httpx.Client | None]:
    params = _client_params(args)
    if args.fresh_connection:
        def factory() -> httpx.Client:
            return httpx.Client(**params)
        shared: httpx.Client | None = None
        close_after = True
    else:
        shared = httpx.Client(**params)

        def factory() -> httpx.Client:
            return shared
        close_after = False

    fetcher = HttpFetcher(factory, close_after_fetch=close_after, cache_bust=args.cache_bust)
    return fetcher, shared


def _ensure_utf8_output() -> None:
    """На Windows stdout/stderr могут быть cp1251/cp437 — переводим на utf-8."""
    for stream in (sys.stdout, sys.stderr):
        if not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except OSError:
            pass


def main(argv: Sequence[str] | None = None) -> int:
    _ensure_utf8_output()
    args = build_parser().parse_args(argv)
    fetcher, shared_client = _build_fetcher(args)

    try:
        stats = SpeedMeter(fetcher, attempts=args.attempts).run(args.url)
    except (httpx.HTTPError, httpx.InvalidURL) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    finally:
        if shared_client is not None:
            shared_client.close()

    ConsoleReporter().report(args.url, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())