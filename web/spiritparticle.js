import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// Displays selected model information on Random Checkpoint Loader and Random LoRA Loader nodes
app.registerExtension({
  name: "SpiritParticle.ModelDisplay",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "RandomCheckpointLoader") {
      function populateCheckpoint(selectedCheckpoint) {
        if (this.widgets) {
          // Remove existing display widgets (preserve input widgets: subfolder, seed)
          const originalWidgetCount = 2;
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

      // When the node is executed we will be sent the selected checkpoint
      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);
        if (message.text && message.text[0]) {
          populateCheckpoint.call(this, message.text[0]);
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
      function populateLoRA(displayData) {
        if (this.widgets) {
          // Remove existing display widgets (preserve input widgets: subfolder, strength_model, strength_clip, seed)
          const originalWidgetCount = 4;
          for (let i = originalWidgetCount; i < this.widgets.length; i++) {
            this.widgets[i].onRemove?.();
          }
          this.widgets.length = originalWidgetCount;
        }

        if (displayData && displayData.length > 0) {
          for (const text of displayData) {
            const displayWidget = ComfyWidgets["STRING"](
              this,
              "lora_display",
              ["STRING", { multiline: true }],
              app
            ).widget;
            displayWidget.inputEl.readOnly = true;
            displayWidget.inputEl.style.opacity = 0.6;
            displayWidget.inputEl.style.backgroundColor = "#2a2a2a";
            displayWidget.value = text;
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

      // When the node is executed we will be sent the selected lora and trigger words
      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);
        if (message.text) {
          populateLoRA.call(this, message.text);
        }
      };

      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        onConfigure?.apply(this, arguments);
        if (this.widgets_values?.length > 4) {
          populateLoRA.call(this, this.widgets_values[4]);
        }
      };
    }
  },
});
