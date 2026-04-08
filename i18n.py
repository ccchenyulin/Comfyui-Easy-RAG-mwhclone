from __future__ import annotations

import json
import locale
from pathlib import Path
from typing import Any, Dict, Optional


_TRANSLATIONS: Dict[str, Dict[str, str]] = {}
LOCALE_PATH = Path(__file__).resolve().parent / "locales"


def _candidate_files(lang: str) -> list[Path]:
    candidates = [
        LOCALE_PATH / lang / "main.json",
    ]
    if "-" in lang:
        base = lang.split("-")[0]
        candidates.extend(
            [
                LOCALE_PATH / base / "main.json",
            ]
        )
    return candidates


def _load_translations(lang: str) -> Dict[str, str]:
    if lang in _TRANSLATIONS:
        return _TRANSLATIONS[lang]

    for p in _candidate_files(lang):
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            _TRANSLATIONS[lang] = data
            return data
        except Exception:
            continue

    if lang != "en":
        for p in _candidate_files("en"):
            if not p.exists():
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                _TRANSLATIONS[lang] = data
                return data
            except Exception:
                continue

    _TRANSLATIONS[lang] = {}
    return {}


def normalize_language(language: Optional[str]) -> str:
    if not language:
        return "en"
    value = str(language).strip().lower()
    if value in {"zh", "zh-cn", "zh-hans", "cn"}:
        return "zh"
    if value in {"zh-tw", "zh-hant"}:
        return "zh-tw"
    if value in {"en", "en-us", "en-gb"}:
        return "en"
    return "en"


def _settings_path() -> Path:
    # Official default settings location in ComfyUI.
    return Path(__file__).resolve().parents[2] / "user" / "default" / "comfy.settings.json"


def _load_settings() -> Dict[str, Any]:
    path = _settings_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def detect_language() -> str:
    """Follow ComfyUI language setting, then fallback to system locale."""
    settings = _load_settings()
    value = settings.get("Comfy.Locale")
    if value:
        return normalize_language(str(value))

    try:
        sys_lang, _ = locale.getlocale()
        if not sys_lang:
            default_locale = locale.getdefaultlocale()
            sys_lang = default_locale[0] if default_locale else None
        return normalize_language(sys_lang)
    except Exception:
        return "en"


def t(text: str, lang: Optional[str] = None, **kwargs: Any) -> str:
    """Translation function using lazy-loaded JSON maps."""
    current = normalize_language(lang or detect_language())
    trans_map = _load_translations(current)
    
    # If key not found in current map, try English map as a second chance
    translated = trans_map.get(text)
    if translated is None and current != "en":
        en_map = _load_translations("en")
        translated = en_map.get(text)
    
    if translated is None:
        translated = text
    
    if kwargs:
        try:
            return translated.format(**kwargs)
        except Exception:
            return translated
    return translated
