"""Входной слой: очистка текста, невидимые символы, длина и лимит частоты.

Слой стоит до агента: любое сообщение пользователя проходит через
``check_message``. Он нормализует Unicode, вырезает невидимые/управляющие
символы, блокирует «контрабанду» из невидимых символов, ограничивает длину
запроса и частоту обращений.
"""

from __future__ import annotations

import math
import re
import threading
import time
import unicodedata
from collections import deque
from dataclasses import dataclass

import config

# Символы, которые не относятся к категории Cf, но тоже визуально пустые.
_EXTRA_INVISIBLE = {
    "\u115f",  # hangul choseong filler
    "\u1160",  # hangul jungseong filler
    "\u17b4",  # khmer vowel inherent aq
    "\u17b5",  # khmer vowel inherent aa
    "\u180e",  # mongolian vowel separator
    "\u2800",  # braille pattern blank
    "\u3164",  # hangul filler
    "\uffa0",  # halfwidth hangul filler
}

# Пробельные символы, которые после NFKC могут остаться необычными.
_SPACE_LIKE = {"\u00a0", "\u2007", "\u202f", "\u2009", "\u200a"}

_WHITESPACE_RUNS = re.compile(r"[ \t]+\n")
_BLANK_LINES = re.compile(r"\n{3,}")


def _is_invisible(char: str) -> bool:
    """True для символов, которые пользователь не видит, а модель — да."""
    if char in _EXTRA_INVISIBLE:
        return True
    category = unicodedata.category(char)
    if category in {"Cf", "Cs", "Co", "Cn"}:
        return True
    if category == "Cc" and char not in "\n\t":
        return True
    code = ord(char)
    # Variation selectors (в том числе supplementary).
    if 0xFE00 <= code <= 0xFE0F or 0xE0100 <= code <= 0xE01EF:
        return True
    return False


def count_invisible(text: str) -> int:
    """Считает невидимые символы, не трогая обычные переводы строк и табы."""
    return sum(
        1
        for char in text
        if char not in "\n\t" and _is_invisible(char)
    )


def normalize_text(text: str) -> str:
    """NFKC + удаление невидимых символов + нормализация пробелов и переводов строк."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned: list[str] = []
    for char in text:
        if char in "\u2028\u2029":  # line/paragraph separator
            cleaned.append("\n")
        elif char in _SPACE_LIKE:
            cleaned.append(" ")
        elif _is_invisible(char):
            continue
        else:
            cleaned.append(char)

    text = "".join(cleaned)
    text = _WHITESPACE_RUNS.sub("\n", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


@dataclass(frozen=True)
class InputDecision:
    """Результат проверки одного сообщения."""

    accepted: bool
    text: str = ""
    reason: str = "ok"  # ok | empty | too_long | invisible | rate_limited
    user_message: str = ""
    removed_invisible: int = 0
    retry_after: int = 0


class InputGuard:
    """Проверки входа с независимыми параметрами (удобно для тестов)."""

    def __init__(
        self,
        *,
        max_chars: int,
        rate_limit: int,
        window_seconds: float,
        invisible_ratio: float,
        invisible_min_chars: int,
        max_keys: int = 10_000,
    ):
        self.max_chars = max(1, int(max_chars))
        self.rate_limit = max(1, int(rate_limit))
        self.window_seconds = max(1.0, float(window_seconds))
        self.invisible_ratio = min(max(float(invisible_ratio), 0.0), 1.0)
        self.invisible_min_chars = max(1, int(invisible_min_chars))
        self.max_keys = max(100, int(max_keys))
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def _rate_limit(self, user_key: str, now: float) -> int:
        """Скользящее окно. Возвращает 0 или сколько секунд ждать."""
        key = str(user_key)
        with self._lock:
            if key not in self._hits and len(self._hits) >= self.max_keys:
                self._evict_oldest()
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self.window_seconds:
                hits.popleft()
            if len(hits) >= self.rate_limit:
                wait = self.window_seconds - (now - hits[0])
                return max(1, math.ceil(wait))
            hits.append(now)
            return 0

    def _evict_oldest(self) -> None:
        oldest_key = min(
            self._hits,
            key=lambda item: self._hits[item][0] if self._hits[item] else float("-inf"),
        )
        self._hits.pop(oldest_key, None)

    def check(
        self,
        raw_text: str,
        user_key: str = "default",
        *,
        now: float | None = None,
    ) -> InputDecision:
        raw_text = raw_text or ""
        if not raw_text.strip():
            return InputDecision(
                accepted=False,
                reason="empty",
                user_message="Пустой запрос. Напишите вопрос текстом.",
            )

        current = time.monotonic() if now is None else now
        retry_after = self._rate_limit(user_key, current)
        if retry_after:
            return InputDecision(
                accepted=False,
                reason="rate_limited",
                user_message=(
                    "Слишком много запросов подряд. "
                    f"Подождите {retry_after} с и повторите."
                ),
                retry_after=retry_after,
            )

        invisible = count_invisible(raw_text)
        if (
            invisible
            and len(raw_text) >= self.invisible_min_chars
            and invisible / len(raw_text) > self.invisible_ratio
        ):
            return InputDecision(
                accepted=False,
                reason="invisible",
                user_message=(
                    "Сообщение содержит слишком много невидимых служебных символов "
                    "и заблокировано. Отправьте обычный текст."
                ),
                removed_invisible=invisible,
            )

        text = normalize_text(raw_text)
        if not text:
            return InputDecision(
                accepted=False,
                reason="empty",
                user_message="После очистки сообщение оказалось пустым.",
                removed_invisible=invisible,
            )

        if len(text) > self.max_chars:
            return InputDecision(
                accepted=False,
                reason="too_long",
                user_message=(
                    f"Слишком длинный запрос: {len(text)} символов при лимите "
                    f"{self.max_chars}. Сократите текст и повторите."
                ),
                removed_invisible=invisible,
            )

        return InputDecision(
            accepted=True,
            text=text,
            removed_invisible=invisible,
        )


GUARD = InputGuard(
    max_chars=config.INPUT_MAX_CHARS,
    rate_limit=config.INPUT_RATE_LIMIT,
    window_seconds=config.INPUT_RATE_WINDOW,
    invisible_ratio=config.INPUT_INVISIBLE_MAX_RATIO,
    invisible_min_chars=config.INPUT_INVISIBLE_MIN_CHARS,
)


def check_message(raw_text: str, user_key: str = "default") -> InputDecision:
    """Проверка сообщения общим (production) слоем."""
    return GUARD.check(raw_text, user_key)
