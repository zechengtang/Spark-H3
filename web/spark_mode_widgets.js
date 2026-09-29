import { app } from "../../../scripts/app.js";

const SPARK_NODE = "MiniMaxH3SparkAttentionSM120";
const MODE_DEFAULTS = {
    warmup_mode: "warmup_percent",
    warmup_steps: 4,
    topk_mode: "topk_ratio",
    topk_blocks: 228,
};

function modeIsLinked(node, name) {
    const input = node.inputs?.find((item) => item.name === name);
    return input?.link != null;
}

function setVisible(widget, visible) {
    if (!widget) return;
    widget.hidden = !visible;
    widget.options ??= {};
    widget.options.hidden = !visible;
}

function syncModeWidgets(node) {
    const widgets = new Map((node.widgets ?? []).map((widget) => [widget.name, widget]));
    const topkMode = widgets.get("topk_mode")?.value ?? "topk_ratio";
    const warmupMode = widgets.get("warmup_mode")?.value ?? "warmup_percent";

    // A linked mode is decided at execution time, so keep both values editable.
    const topkLinked = modeIsLinked(node, "topk_mode");
    const warmupLinked = modeIsLinked(node, "warmup_mode");
    setVisible(widgets.get("topk_ratio"), topkLinked || topkMode === "topk_ratio");
    setVisible(widgets.get("topk_blocks"), topkLinked || topkMode === "topk_blocks");
    setVisible(widgets.get("warmup_percent"), warmupLinked || warmupMode === "warmup_percent");
    setVisible(widgets.get("warmup_steps"), warmupLinked || warmupMode === "warmup_steps");

    if (typeof node.computeSize === "function" && typeof node.setSize === "function") {
        const [width, height] = node.computeSize();
        node.setSize([Math.max(node.size[0], width), height]);
    }
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "MiniMaxH3.SparkModeWidgets",
    nodeCreated(node) {
        if (node.comfyClass !== SPARK_NODE && node.constructor?.comfyClass !== SPARK_NODE) return;
        if (node._sparkModeWidgetsInstalled) return;
        node._sparkModeWidgetsInstalled = true;

        for (const name of ["topk_mode", "warmup_mode"]) {
            const widget = node.widgets?.find((item) => item.name === name);
            if (!widget) continue;
            const previousCallback = widget.callback;
            widget.callback = function (...args) {
                const result = previousCallback?.apply(this, args);
                syncModeWidgets(node);
                return result;
            };
        }

        const previousConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function (...args) {
            const result = previousConnectionsChange?.apply(this, args);
            syncModeWidgets(this);
            return result;
        };
        const previousConfigure = node.onConfigure;
        node.onConfigure = function (serialized) {
            const result = previousConfigure?.call(this, serialized);
            // Older workflows can restore widget values by position after the
            // schema order changes. Use their saved names to recover the values.
            const saved = serialized?.widgets_values_named;
            if (saved) {
                for (const widget of this.widgets ?? []) {
                    if (Object.prototype.hasOwnProperty.call(saved, widget.name)) {
                        widget.value = saved[widget.name];
                    } else if (Object.prototype.hasOwnProperty.call(MODE_DEFAULTS, widget.name)) {
                        widget.value = MODE_DEFAULTS[widget.name];
                    }
                }
            }
            syncModeWidgets(this);
            return result;
        };
        syncModeWidgets(node);
    },
    loadedGraphNode(node) {
        if (node.comfyClass === SPARK_NODE || node.constructor?.comfyClass === SPARK_NODE) {
            syncModeWidgets(node);
        }
    },
});
