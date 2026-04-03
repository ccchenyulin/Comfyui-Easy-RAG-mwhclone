import { app } from "/scripts/app.js";

const TRANSLATIONS = {
  en: {
    "上传文档": "Upload document",
    "文档上传失败: {error}": "Document upload failed: {error}",
    "Uploaded document": "Uploaded document",
  },
  zh: {
    "上传文档": "上传文档",
    "文档上传失败: {error}": "文档上传失败: {error}",
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
    if (raw.includes("zh") || raw.includes("cn")) return "zh";
    if (raw.includes("en")) return "en";
  }
  return "zh"; // Default to zh based on the user's preference
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
