import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// Displays input text on a node
app.registerExtension({
  name: "ShowText",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData.name === "ShowText") {
      function populate(text) {
        const v = [...text];
        if (!v[0]) {
          v.shift();
        }
        
        // Find existing display widget or create one
        let displayWidget = this.widgets?.find(w => w.name === "text2");
        if (!displayWidget) {
          displayWidget = ComfyWidgets["STRING"](
            this,
            "text2",
            ["STRING", { multiline: true }],
            app
          ).widget;
          displayWidget.inputEl.readOnly = true;
          displayWidget.inputEl.style.opacity = 0.6;
        }
        
        // Update the widget value with the first text item
        displayWidget.value = v[0] || "";

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

      // When the node is executed we will be sent the input text, display this in the widget
      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);
        populate.call(this, message.text);
      };

      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        onConfigure?.apply(this, arguments);
        if (this.widgets_values?.length) {
          populate.call(
            this,
            this.widgets_values.slice(+this.widgets_values.length > 1)
          );
        }
      };
    }
  },
});
