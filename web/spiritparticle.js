import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// Displays selected model information on Random Checkpoint Loader and Random LoRA Loader nodes
app.registerExtension({
  name: "SpiritParticle.ModelDisplay",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "RandomCheckpointLoader") {
      function populateCheckpoint(selectedCheckpoint) {
        if (this.widgets) {
          // Remove existing display widgets (keep original widgets)
          const originalWidgetCount = 2; // subfolder and seed
          for (let i = originalWidgetCount; i < this.widgets.length; i++) {
            this.widgets[i].onRemove?.();
          }
          this.widgets.length = originalWidgetCount;
        }

        if (selectedCheckpoint) {
          const displayWidget = ComfyWidgets["STRING"](
            this,
            "selected_checkpoint_display",
            ["STRING", { multiline: false }],
            app
          ).widget;
          displayWidget.inputEl.readOnly = true;
          displayWidget.inputEl.style.opacity = 0.6;
          displayWidget.inputEl.style.fontWeight = "bold";
          displayWidget.inputEl.style.backgroundColor = "#2a2a2a";
          displayWidget.value = `Selected: ${selectedCheckpoint}`;
        }

        requestAnimationFrame(() => {
          const sz = this.computeSize();
          if (sz[0] < this.size[0]) {
            sz[0] = this.size[0];
          }
          if (sz[1] < this.size[1]) {
            sz[1] = this.size[1];
          }
          this.onResize?.(sz);
          app.graph.setDirtyCanvas(true, false);
        });
      }

      // When the node is executed, display the selected checkpoint
      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);
        if (message.selected_checkpoint) {
          populateCheckpoint.call(this, message.selected_checkpoint[0]);
        }
      };

      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        onConfigure?.apply(this, arguments);
        if (this.widgets_values?.length > 2) {
          populateCheckpoint.call(this, this.widgets_values[2]);
        }
      };
    }

    if (nodeData.name === "RandomLoRALoader") {
      function populateLoRA(selectedLora, triggerWords) {
        if (this.widgets) {
          // Remove existing display widgets (keep original widgets)
          const originalWidgetCount = 6; // model, clip, subfolder, strength_model, strength_clip, seed
          for (let i = originalWidgetCount; i < this.widgets.length; i++) {
            this.widgets[i].onRemove?.();
          }
          this.widgets.length = originalWidgetCount;
        }

        if (selectedLora) {
          // Display selected LoRA
          const loraDisplayWidget = ComfyWidgets["STRING"](
            this,
            "selected_lora_display",
            ["STRING", { multiline: false }],
            app
          ).widget;
          loraDisplayWidget.inputEl.readOnly = true;
          loraDisplayWidget.inputEl.style.opacity = 0.6;
          loraDisplayWidget.inputEl.style.fontWeight = "bold";
          loraDisplayWidget.inputEl.style.backgroundColor = "#2a2a2a";
          loraDisplayWidget.value = `Selected: ${selectedLora}`;

          // Display trigger words if available
          if (triggerWords && triggerWords.trim()) {
            const triggerDisplayWidget = ComfyWidgets["STRING"](
              this,
              "trigger_words_display",
              ["STRING", { multiline: true }],
              app
            ).widget;
            triggerDisplayWidget.inputEl.readOnly = true;
            triggerDisplayWidget.inputEl.style.opacity = 0.6;
            triggerDisplayWidget.inputEl.style.fontStyle = "italic";
            triggerDisplayWidget.inputEl.style.backgroundColor = "#2a2a2a";
            triggerDisplayWidget.value = `Triggers: ${triggerWords}`;
          }
        }

        requestAnimationFrame(() => {
          const sz = this.computeSize();
          if (sz[0] < this.size[0]) {
            sz[0] = this.size[0];
          }
          if (sz[1] < this.size[1]) {
            sz[1] = this.size[1];
          }
          this.onResize?.(sz);
          app.graph.setDirtyCanvas(true, false);
        });
      }

      // When the node is executed, display the selected LoRA and trigger words
      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);
        if (message.selected_lora) {
          const selectedLora = message.selected_lora[0];
          const triggerWords = message.trigger_words ? message.trigger_words[0] : "";
          populateLoRA.call(this, selectedLora, triggerWords);
        }
      };

      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        onConfigure?.apply(this, arguments);
        if (this.widgets_values?.length > 6) {
          const selectedLora = this.widgets_values[6];
          const triggerWords = this.widgets_values.length > 7 ? this.widgets_values[7] : "";
          populateLoRA.call(this, selectedLora, triggerWords);
        }
      };
    }
  },
});
