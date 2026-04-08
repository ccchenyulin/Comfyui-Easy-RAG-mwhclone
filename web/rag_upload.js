import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { t } from "./i18n.js";

const TARGET_NODE_NAMES = new Set([
  "RagPromptDocumentLoader",
]);
const UPLOAD_WIDGET_FLAG = "__easy_rag_upload_button";
const DOC_WIDGET_NAMES = ["document", "文档"];

function hasChineseUI(node) {
  return (node.widgets || []).some((w) => /[\u3400-\u9fff]/.test(String(w?.name || "")));
}

function getWidgetByNames(node, names) {
  const set = new Set(names.map((x) => String(x || "").trim().toLowerCase()));
  return (node.widgets || []).find((w) => set.has(String(w?.name || "").trim().toLowerCase()));
}

function getComboValues(widget) {
  if (!widget) return [];
  if (Array.isArray(widget.options?.values)) return widget.options.values;
  if (Array.isArray(widget.options)) return widget.options;
  return [];
}

function setComboValues(widget, values) {
  if (!widget) return;
  if (widget.options && Array.isArray(widget.options.values)) {
    widget.options.values = values;
    return;
  }
  if (widget.options && !Array.isArray(widget.options.values)) {
    widget.options.values = values;
    return;
  }
  widget.options = values;
}

app.registerExtension({
  name: "rag.document-upload",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!TARGET_NODE_NAMES.has(nodeData.name)) {
      return;
    }

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;

      const hasButton = (this.widgets || []).some((w) => !!w?.[UPLOAD_WIDGET_FLAG]);
      if (hasButton) return r;

      const self = this;
      const buttonLabel = hasChineseUI(this) ? "上传文件夹" : t("Upload Folder");
      const uploadWidget = this.addWidget("button", buttonLabel, buttonLabel, async () => {
        const picker = document.createElement("input");
        picker.type = "file";
        picker.accept = ".txt,.json,.md,.pdf";
        picker.style.display = "none";

        picker.onchange = async () => {
          const file = picker.files && picker.files[0];
          if (!file) return;

          try {
            const formData = new FormData();
            formData.append("image", file);
            formData.append("type", "input");

            const resp = await api.fetchApi("/upload/image", {
              method: "POST",
              body: formData,
            });
            if (!resp.ok) {
              const text = await resp.text();
              throw new Error(text || `HTTP ${resp.status}`);
            }
            const data = await resp.json();
            const savedName = data?.name || file.name;

            const docWidget = getWidgetByNames(self, DOC_WIDGET_NAMES);
            if (docWidget) {
              const values = getComboValues(docWidget).slice();
              if (!values.includes(savedName)) {
                values.push(savedName);
                values.sort();
                setComboValues(docWidget, values);
              }
              docWidget.value = savedName;
            }

            if (app.graph) {
              app.graph.setDirtyCanvas(true, true);
            }
          } catch (err) {
            console.error("[Rag] 文档上传失败:", err);
            alert(t("Document upload failed: {error}", { error: err?.message || err }));
          } finally {
            picker.remove();
          }
        };

        document.body.appendChild(picker);
        picker.click();
      });
      if (uploadWidget) uploadWidget[UPLOAD_WIDGET_FLAG] = true;

      return r;
    };
  },
});
