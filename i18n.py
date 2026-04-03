from __future__ import annotations

import json
import locale
import os
from pathlib import Path
from typing import Any, Dict, Optional


LANGUAGE_SETTING_ID = "EasyRAG.Language"
DEFAULT_LANGUAGE = "auto"

_TRANSLATIONS: Dict[str, Dict[str, str]] = {}
_LOCALES_DIR = Path(__file__).resolve().parent / "locales"

def _load_translations(lang: str) -> Dict[str, str]:
    if lang in _TRANSLATIONS:
        return _TRANSLATIONS[lang]
    
    # Check current lang folder for messages.json
    p = _LOCALES_DIR / lang / "messages.json"
    if not p.exists():
        # Fallback for zh-CN -> zh etc
        if "-" in lang:
            base = lang.split("-")[0]
            p = _LOCALES_DIR / base / "messages.json"
    
    if not p.exists() and lang != "en":
        # Final fallback to en
        p = _LOCALES_DIR / "en" / "messages.json"

    if not (p and p.exists()):
        return {}

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        _TRANSLATIONS[lang] = data
        return data
    except Exception:
        return {}


def normalize_language(language: Optional[str]) -> str:
    if not language:
        return "en"
    value = str(language).strip().lower()
    if value in {"zh", "zh-cn", "zh-hans", "cn"}:
        return "zh"
    if value in {"en", "en-us", "en-gb"}:
        return "en"
    return "en"


def _settings_path() -> Path:
    # Look for comfy.settings.json in standard location
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
    """Detection based on official ComfyUI settings only."""
    settings = _load_settings()
    value = settings.get("Comfy.Locale")
    if value:
        return normalize_language(str(value))
    
    # Fallback to system locale if no setting
    try:
        sys_lang, _ = locale.getlocale()
        if not sys_lang:
            sys_lang = locale.getdefaultlocale()[0] if locale.getdefaultlocale() else None
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
