// A custom card for Home Assistant to enroll a person with face images.
const DEFAULT_TITLE = "Enroll Person";
const DEFAULT_ENDPOINT = "/api/home_generative_agent/enroll";

class HgaEnrollCard extends HTMLElement {
  setConfig(config) {
    if (!config) {
      throw new Error("Invalid configuration");
    }
    this._config = {
      title: DEFAULT_TITLE,
      endpoint: DEFAULT_ENDPOINT,
      ...config,
    };
    this._render();
  }

  set hass(hass) {
    this._hass = hass;
  }

  getCardSize() {
    return 3;
  }

  _render() {
    if (!this.shadowRoot) {
      this.attachShadow({ mode: "open" });
    }
    this.shadowRoot.innerHTML = `
      <style>
        .row { margin: 10px 0; }
        .status { min-height: 18px; font-size: 12px; color: var(--secondary-text-color); }
        .status.error { color: var(--error-color); }
        .status.success { color: var(--success-color); }
        input[type="text"] { width: 100%; box-sizing: border-box; }
        button { padding: 8px 12px; }
        .dropzone {
          border: 1px dashed var(--divider-color);
          border-radius: 6px;
          padding: 12px;
          text-align: center;
          color: var(--secondary-text-color);
        }
        .dropzone.dragover {
          border-color: var(--primary-color);
          color: var(--primary-color);
        }
        .file-list { font-size: 12px; color: var(--secondary-text-color); }
      </style>
      <ha-card header="${this._config.title}">
        <div class="card-content">
          <div class="row">
            <input id="name" type="text" placeholder="Person name" />
          </div>
          <div class="row">
            <input id="file" type="file" accept="image/*" multiple />
            <div id="dropzone" class="dropzone">Drop images here</div>
            <div id="fileList" class="file-list"></div>
            <button id="clear" type="button">Clear files</button>
          </div>
          <div class="row">
            <button id="enroll">Enroll</button>
          </div>
          <div id="status" class="status"></div>
        </div>
      </ha-card>
    `;

    this._nameInput = this.shadowRoot.getElementById("name");
    this._fileInput = this.shadowRoot.getElementById("file");
    this._button = this.shadowRoot.getElementById("enroll");
    this._status = this.shadowRoot.getElementById("status");
    this._dropzone = this.shadowRoot.getElementById("dropzone");
    this._fileList = this.shadowRoot.getElementById("fileList");
    this._clearButton = this.shadowRoot.getElementById("clear");
    this._files = [];

    this._fileInput.addEventListener("change", () => {
      this._addFiles(this._fileInput.files);
      this._fileInput.value = "";
    });
    this._dropzone.addEventListener("dragover", (event) => {
      event.preventDefault();
      this._dropzone.classList.add("dragover");
    });
    this._dropzone.addEventListener("dragleave", () => {
      this._dropzone.classList.remove("dragover");
    });
    this._dropzone.addEventListener("drop", (event) => {
      event.preventDefault();
      this._dropzone.classList.remove("dragover");
      this._addFiles(event.dataTransfer.files);
    });
    this._clearButton.addEventListener("click", () => {
      this._files = [];
      this._renderFileList();
    });
    this._button.addEventListener("click", () => this._enroll());
  }

  _addFiles(fileList) {
    const files = Array.from(fileList || []);
    for (const file of files) {
      if (file.type && !file.type.startsWith("image/")) {
        this._setStatus(`Skipped non-image: ${file.name}`, "error");
        continue;
      }
      this._files.push(file);
    }
    this._renderFileList();
  }

  _renderFileList() {
    if (!this._fileList) {
      return;
    }
    if (this._files.length === 0) {
      this._fileList.textContent = "No files selected.";
      return;
    }
    const names = this._files.map((file) => file.name).join(", ");
    this._fileList.textContent = `${this._files.length} file(s): ${names}`;
  }

  _setStatus(message, kind) {
    this._status.textContent = message || "";
    this._status.className = `status${kind ? ` ${kind}` : ""}`;
  }

  async _enroll() {
    const name = (this._nameInput.value || "").trim();
    const files = this._files || [];

    if (!name) {
      this._setStatus("Name is required.", "error");
      return;
    }
    if (files.length === 0) {
      this._setStatus("Image is required.", "error");
      return;
    }

    this._button.disabled = true;
    this._setStatus("Uploading...", "");

    try {
      const formData = new FormData();
      formData.append("name", name);
      for (const file of files) {
        formData.append("file", file, file.name);
      }

      const token =
        (this._hass.auth && this._hass.auth.data && this._hass.auth.data.access_token) ||
        (this._hass.auth && this._hass.auth.accessToken);
      const url = this._hass.hassUrl(this._config.endpoint);

      const response = await fetch(url, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const message = data.message || `Upload failed (${response.status}).`;
        this._setStatus(message, "error");
        return;
      }
      this._setStatus(data.message || "Enrolled.", "success");
      this._files = [];
      this._renderFileList();
    } catch (err) {
      this._setStatus(`Upload failed: ${err}`, "error");
    } finally {
      this._button.disabled = false;
    }
  }
}

customElements.define("hga-enroll-card", HgaEnrollCard);

window.customCards = window.customCards || [];
window.customCards.push({
  type: "hga-enroll-card",
  name: "HGA Enroll Person",
  description: "Upload a face image and enroll a person.",
});
