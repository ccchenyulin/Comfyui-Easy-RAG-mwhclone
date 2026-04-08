import { app } from "/scripts/app.js";

const TRANSLATIONS = {
  en: {
    "Upload document": "Upload document",
    "Document upload failed: {error}": "Document upload failed: {error}",
    "Uploaded document": "Uploaded document",
  },
  zh: {
    "Upload document": "上传文件夹",
    "Document upload failed: {error}": "文档上传失败: {error}",
    "Uploaded document": "已上传文档",
  },
  "zh-tw": {
    "Upload document": "上傳資料夾",
    "Document upload failed: {error}": "文件上傳失敗: {error}",
    "Uploaded document": "已上传文档",
  },
};

/**
 * Gets the current language based on official ComfyUI settings.
 */
function getLanguage() {
  const comfyLocale = app.ui.settings.getSettingValue("Comfy.Locale");
  if (comfyLocale) {
    const raw = String(comfyLocale).toLowerCase();
    if (raw.includes("zh-tw") || raw.includes("zh-hant")) return "zh-tw";
    if (raw.includes("zh") || raw.includes("cn")) return "zh";
    if (raw.includes("en")) return "en";
  }
  const browserLocale = String(globalThis?.navigator?.language || "").toLowerCase();
  if (browserLocale.includes("zh-tw") || browserLocale.includes("zh-hant")) return "zh-tw";
  if (browserLocale.includes("zh")) return "zh";
  return "en";
}

/**
 * Translates a key with optional dynamic values.
 * @param {string} text - The key to translate.
 * @param {object} options - Optional values for {bracket} placeholders.
 * @returns {string}
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
