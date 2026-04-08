import { app } from "/scripts/app.js";

const TRANSLATIONS = {
  en: {
    "Upload Folder": "Upload Folder",
    "Document upload failed: {error}": "Document upload failed: {error}",
    "Uploaded document": "Uploaded document",
  },
  zh: {
    "Upload Folder": "上传文件夹",
    "Document upload failed: {error}": "文档上传失败: {error}",
    "Uploaded document": "已上传文档",
  },
  "zh-tw": {
    "Upload Folder": "上傳資料夾",
    "Document upload failed: {error}": "文件上傳失敗: {error}",
    "Uploaded document": "已上传文档",
  },
};

/**
 * Gets the current language based on official ComfyUI settings.
 */
function getLanguage() {
  const normalize = (value) => {
    const raw = String(value || "").toLowerCase();
    if (!raw) return "";
    if (raw.includes("zh-tw") || raw.includes("zh-hant")) return "zh-tw";
    if (raw.includes("zh") || raw.includes("cn")) return "zh";
    if (raw.includes("en")) return "en";
    return "";
  };

  try {
    const comfyLocale = app?.ui?.settings?.getSettingValue?.("Comfy.Locale");
    const normalized = normalize(comfyLocale);
    if (normalized) return normalized;
  } catch (_) {}

  try {
    const managerLocale = app?.extensionManager?.settingStore?.get?.("Comfy.Locale");
    const normalized = normalize(managerLocale);
    if (normalized) return normalized;
  } catch (_) {}

  try {
    const domLang = normalize(globalThis?.document?.documentElement?.lang);
    if (domLang) return domLang;
  } catch (_) {}

  try {
    const storageLocale = normalize(globalThis?.localStorage?.getItem?.("Comfy.Settings.Comfy.Locale"));
    if (storageLocale) return storageLocale;
  } catch (_) {}

  try {
    const storageLocaleAlt = normalize(globalThis?.localStorage?.getItem?.("Comfy.Locale"));
    if (storageLocaleAlt) return storageLocaleAlt;
  } catch (_) {}

  try {
    const browserLocale = String(globalThis?.navigator?.language || "").toLowerCase();
    if (browserLocale.includes("zh-tw") || browserLocale.includes("zh-hant")) return "zh-tw";
    if (browserLocale.includes("zh")) return "zh";
  } catch (_) {}

  // Prefer English only as the final fallback.
  return "en";
}

/**
 * Returns translated text.
 */
export function t(text, options = {}) {
  const lang = getLanguage();
  let translated = TRANSLATIONS[lang]?.[text] || TRANSLATIONS["en"]?.[text] || text;

  const values = options?.values || options;
  if (values) {
    for (const [k, v] of Object.entries(values)) {
      translated = translated.replace(`{${k}}`, v);
    }
  }

  return translated;
}
