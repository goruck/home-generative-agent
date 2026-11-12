const HGA_MEMORY_I18N = {
  en: {
    refresh: "Refresh",
    search: "Search memories",
    shared: "Shared (unknown voice)",
    empty: "No memories yet.",
    empty_person: "No memories for this person.",
    no_match: "Nothing matches the search.",
    limit: "{n} of {max}",
    updated: "Changed {when}",
    just_now: "just now",
    edit: "Edit",
    save: "Save",
    cancel: "Cancel",
    move: "Move to",
    remove: "Delete",
    confirm_remove: "Delete this memory?",
    failed: "Failed: {msg}",
    admin_only: "Only an administrator can manage memories.",
    title_label: "Title",
    title_helper: "Leave empty to show no title.",
    person_label: "Start with",
    person_helper: "Leave empty to start with the first person who has memories.",
    show_empty_label: "Show people with no memories",
    show_search_label: "Show search",
  },
  ru: {
    refresh: "Обновить",
    search: "Поиск воспоминаний",
    shared: "Общие (неизвестный голос)",
    empty: "Воспоминаний пока нет.",
    empty_person: "У этого человека нет воспоминаний.",
    no_match: "Ничего не найдено.",
    limit: "{n} из {max}",
    updated: "Изменено {when}",
    just_now: "только что",
    edit: "Изменить",
    save: "Сохранить",
    cancel: "Отмена",
    move: "Переместить к",
    remove: "Удалить",
    confirm_remove: "Удалить это воспоминание?",
    failed: "Ошибка: {msg}",
    admin_only: "Управлять воспоминаниями может только администратор.",
    title_label: "Заголовок",
    title_helper: "Оставьте пустым, чтобы не показывать заголовок.",
    person_label: "Начинать с",
    person_helper: "Оставьте пустым, чтобы начинать с первого человека с воспоминаниями.",
    show_empty_label: "Показывать людей без воспоминаний",
    show_search_label: "Показывать поиск",
  },
};

const HGA_SHARED_MEMORY_USER = "robot";

const hgaMemoryLanguage = (hass) =>
  (hass?.locale?.language || hass?.language || "en").split("-")[0];

const hgaMemoryText = (hass, key, values = {}) => {
  const text = (HGA_MEMORY_I18N[hgaMemoryLanguage(hass)] || HGA_MEMORY_I18N.en)[key];
  return Object.entries(values).reduce(
    (out, [name, value]) => out.replace(`{${name}}`, value),
    text,
  );
};

const hgaMemoryLabel = (hass, user) =>
  user === HGA_SHARED_MEMORY_USER ? hgaMemoryText(hass, "shared") : user;

const hgaMemoryCall = async (hass, data) => {
  const result = await hass.callWS({
    type: "call_service",
    domain: "home_generative_agent",
    service: data.service,
    service_data: data.data,
    return_response: true,
  });
  return result.response;
};

const HGA_MEMORY_STYLE = `
  .wrap { padding: 16px; display: flex; flex-direction: column; gap: 12px; }
  h2 { margin: 0; font-size: 20px; font-weight: 500; line-height: 28px; }
  .top { display: flex; align-items: flex-start; gap: 8px; }
  .people { display: flex; flex: 1; min-width: 0; flex-wrap: wrap; gap: 8px; }
  .chip { display: inline-flex; align-items: center; gap: 8px; max-width: 100%; min-height: 36px;
    padding: 0 12px; border-radius: 18px; border: 1px solid var(--divider-color); background: transparent;
    color: var(--primary-text-color); font: inherit; font-size: 14px; cursor: pointer; }
  .chip:hover { background: var(--secondary-background-color); }
  .chip.active { border-color: var(--primary-color); color: var(--primary-color);
    background: color-mix(in srgb, var(--primary-color) 16%, transparent); }
  .chip .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .chip .count { font-size: 12px; color: var(--secondary-text-color); }
  .chip.active .count { color: inherit; }
  .search { position: relative; color: var(--secondary-text-color); }
  .search ha-icon { position: absolute; left: 10px; top: 8px; --mdc-icon-size: 20px; pointer-events: none; }
  .search input { width: 100%; height: 36px; padding: 0 12px 0 38px; }
  input, textarea { box-sizing: border-box; font: inherit; font-size: 14px; color: var(--primary-text-color);
    background: var(--secondary-background-color); border: 1px solid transparent; border-radius: 8px; outline: none; }
  input:focus, textarea:focus { border-color: var(--primary-color); }
  textarea { width: 100%; padding: 10px 12px; resize: vertical; line-height: 20px; }
  .list { display: flex; flex-direction: column; }
  .memory { display: flex; align-items: flex-start; gap: 8px; padding: 12px 0;
    border-top: 1px solid var(--divider-color); }
  .memory.editing { flex-direction: column; align-items: stretch; }
  .body { flex: 1; min-width: 0; }
  .text { white-space: pre-wrap; overflow-wrap: anywhere; font-size: 14px; line-height: 20px; }
  .meta { color: var(--secondary-text-color); font-size: 12px; line-height: 16px; margin-top: 4px; }
  .actions { display: flex; flex: none; align-items: center; }
  .icon-button { position: relative; display: inline-flex; align-items: center; justify-content: center;
    width: 36px; height: 36px; padding: 0; border: 0; border-radius: 50%; background: transparent;
    color: var(--secondary-text-color); cursor: pointer; --mdc-icon-size: 20px; }
  .icon-button:hover { background: var(--secondary-background-color); color: var(--primary-text-color); }
  .icon-button.danger:hover { color: var(--error-color); }
  .icon-button.busy { opacity: 0.5; pointer-events: none; }
  .icon-button select { position: absolute; inset: 0; width: 100%; height: 100%; opacity: 0; cursor: pointer; }
  .buttons { display: flex; align-items: center; justify-content: flex-end; gap: 8px; }
  .buttons .prompt { flex: 1; font-size: 14px; color: var(--primary-text-color); }
  .memory.confirming { flex-direction: column; align-items: stretch; gap: 8px; }
  .button { min-height: 36px; padding: 0 14px; border: 0; border-radius: 18px; font: inherit; font-size: 14px;
    font-weight: 500; cursor: pointer; background: transparent; color: var(--primary-color); }
  .button:hover { background: var(--secondary-background-color); }
  .button.primary { background: var(--primary-color); color: var(--text-primary-color); }
  .button.danger { background: var(--error-color); color: var(--text-primary-color, #fff); }
  .message { display: flex; flex-direction: column; align-items: center; gap: 8px; padding: 24px 0;
    color: var(--secondary-text-color); font-size: 14px; --mdc-icon-size: 32px; }
  .error { color: var(--error-color); font-size: 14px; }
  .error:empty { display: none; }
  button:focus-visible, select:focus-visible { outline: 2px solid var(--primary-color); outline-offset: 1px; }
`;

class HgaMemoryCard extends HTMLElement {
  static getConfigElement() {
    return document.createElement("hga-memory-card-editor");
  }

  static getStubConfig() {
    return {};
  }

  setConfig(config) {
    const person = this._config?.person;
    this._config = config || {};
    if (this._config.person !== person) {
      this._user = undefined;
    }
    if (this._started) {
      this._build();
      this._load();
    }
  }

  getCardSize() {
    return 4;
  }

  getGridOptions() {
    return { columns: "full", min_columns: 6 };
  }

  set hass(hass) {
    this._hass = hass;
    if (!this._started) {
      this._started = true;
      this._build();
      this._load();
    }
  }

  _t(key, values) {
    return hgaMemoryText(this._hass, key, values);
  }

  _el(tag, props = {}, ...children) {
    const node = Object.assign(document.createElement(tag), props);
    node.append(...children);
    return node;
  }

  _icon(icon) {
    return this._el("ha-icon", { icon });
  }

  _iconButton(icon, label, onclick, className = "") {
    return this._el(
      "button",
      { className: `icon-button ${className}`, title: label, ariaLabel: label, onclick },
      this._icon(icon),
    );
  }

  _button(textContent, onclick, className = "") {
    return this._el("button", { className: `button ${className}`, textContent, onclick });
  }

  _call(service, data) {
    return hgaMemoryCall(this._hass, { service, data });
  }

  _build() {
    const root = this.shadowRoot || this.attachShadow({ mode: "open" });
    const wanted = this._search?.value || "";
    this._people = this._el("div", { className: "people" });
    this._list = this._el("div", { className: "list" });
    this._error = this._el("div", { className: "error", role: "alert" });
    this._search = this._el("input", {
      type: "search",
      value: wanted,
      placeholder: this._t("search"),
      ariaLabel: this._t("search"),
      oninput: () => this._renderList(),
    });
    this._searchBox = this._el(
      "div",
      { className: "search" },
      this._icon("mdi:magnify"),
      this._search,
    );
    this._refresh = this._iconButton("mdi:refresh", this._t("refresh"), () =>
      this._load(),
    );
    const parts = [];
    if (this._config.title) {
      parts.push(this._el("h2", { textContent: this._config.title }));
    }
    parts.push(
      this._el("div", { className: "top" }, this._people, this._refresh),
      this._searchBox,
      this._error,
      this._list,
    );
    root.replaceChildren(
      this._el("style", { textContent: HGA_MEMORY_STYLE }),
      this._el("ha-card", {}, this._el("div", { className: "wrap" }, ...parts)),
    );
  }

  async _guard(action) {
    this._error.textContent = "";
    try {
      await action();
    } catch (err) {
      this._error.textContent =
        err?.code === "unauthorized"
          ? this._t("admin_only")
          : this._t("failed", { msg: err?.message || String(err) });
    }
  }

  _pickUser(people) {
    const selectable = (user) =>
      people.some((p) => p.user === user && (p.count || this._config.show_empty));
    if (selectable(this._user)) {
      return this._user;
    }
    if (selectable(this._config.person)) {
      return this._config.person;
    }
    const first = people.find((p) => p.count) || (this._config.show_empty && people[0]);
    return first ? first.user : undefined;
  }

  async _load() {
    this._refresh.classList.add("busy");
    await this._guard(async () => {
      let data = await this._call("get_memories", {});
      this._user = this._pickUser(data.people);
      if (this._user) {
        data = await this._call("get_memories", { user: this._user });
      }
      this._data = data;
      this._editing = null;
      this._confirming = null;
      this._renderPeople();
      this._renderList();
    });
    this._refresh.classList.remove("busy");
  }

  _renderPeople() {
    const { people, limit } = this._data;
    const shown = this._config.show_empty ? people : people.filter((p) => p.count);
    this._people.replaceChildren(
      ...shown.map((person) => {
        const count = limit
          ? this._t("limit", { n: person.count, max: limit })
          : String(person.count);
        return this._el(
          "button",
          {
            className: `chip ${person.user === this._user ? "active" : ""}`,
            ariaPressed: String(person.user === this._user),
            onclick: () => {
              this._user = person.user;
              this._load();
            },
          },
          this._el("span", {
            className: "name",
            textContent: hgaMemoryLabel(this._hass, person.user),
          }),
          this._el("span", { className: "count", textContent: count }),
        );
      }),
    );
  }

  _when(iso) {
    const seconds = (new Date(iso).getTime() - Date.now()) / 1000;
    const format = new Intl.RelativeTimeFormat(hgaMemoryLanguage(this._hass), {
      numeric: "auto",
    });
    for (const [unit, size] of [
      ["year", 31536000],
      ["month", 2592000],
      ["day", 86400],
      ["hour", 3600],
      ["minute", 60],
    ]) {
      if (Math.abs(seconds) >= size) {
        return format.format(Math.trunc(seconds / size), unit);
      }
    }
    return this._t("just_now");
  }

  _message(icon, key) {
    return this._el(
      "div",
      { className: "message" },
      this._icon(icon),
      this._t(key),
    );
  }

  _renderList() {
    const memories = this._data?.memories || [];
    const searchable = memories.length && this._config.show_search !== false;
    this._searchBox.style.display = searchable ? "" : "none";
    const wanted = searchable ? this._search.value.trim().toLowerCase() : "";
    const shown = memories.filter((m) =>
      `${m.content} ${m.context}`.toLowerCase().includes(wanted),
    );
    if (shown.length) {
      this._list.replaceChildren(...shown.map((m) => this._renderMemory(m)));
      return;
    }
    if (memories.length) {
      this._list.replaceChildren(this._message("mdi:magnify-close", "no_match"));
      return;
    }
    const key = this._user && this._config.show_empty ? "empty_person" : "empty";
    this._list.replaceChildren(this._message("mdi:brain", key));
  }

  _renderMemory(memory) {
    if (memory.id === this._editing) {
      return this._editor(memory);
    }
    return memory.id === this._confirming ? this._confirm(memory) : this._row(memory);
  }

  _setMode(editing, confirming) {
    this._editing = editing;
    this._confirming = confirming;
    this._renderList();
  }

  _body(memory) {
    const parts = [this._el("div", { className: "text", textContent: memory.content })];
    if (memory.context) {
      parts.push(this._el("div", { className: "meta", textContent: memory.context }));
    }
    parts.push(
      this._el("div", {
        className: "meta",
        textContent: this._t("updated", { when: this._when(memory.updated_at) }),
        title: new Date(memory.updated_at).toLocaleString(hgaMemoryLanguage(this._hass)),
      }),
    );
    return this._el("div", { className: "body" }, ...parts);
  }

  _moveControl(memory) {
    const user = this._user;
    const label = this._t("move");
    const select = this._el(
      "select",
      {
        ariaLabel: label,
        onchange: (event) =>
          this._guard(async () => {
            await this._call("move_memory", {
              user,
              memory_id: memory.id,
              to_user: event.target.value,
            });
            await this._load();
          }),
      },
      this._el("option", {
        textContent: label,
        value: "",
        selected: true,
        disabled: true,
      }),
      ...this._data.people
        .filter((p) => p.user !== user)
        .map((p) =>
          this._el("option", {
            textContent: hgaMemoryLabel(this._hass, p.user),
            value: p.user,
          }),
        ),
    );
    return this._el(
      "label",
      { className: "icon-button", title: label },
      this._icon("mdi:account-arrow-right-outline"),
      select,
    );
  }

  _row(memory) {
    return this._el(
      "div",
      { className: "memory" },
      this._body(memory),
      this._el(
        "div",
        { className: "actions" },
        this._iconButton("mdi:pencil-outline", this._t("edit"), () =>
          this._setMode(memory.id, null),
        ),
        this._moveControl(memory),
        this._iconButton(
          "mdi:delete-outline",
          this._t("remove"),
          () => this._setMode(null, memory.id),
          "danger",
        ),
      ),
    );
  }

  _confirm(memory) {
    return this._el(
      "div",
      { className: "memory confirming" },
      this._body(memory),
      this._el(
        "div",
        { className: "buttons" },
        this._el("span", { className: "prompt", textContent: this._t("confirm_remove") }),
        this._button(this._t("cancel"), () => this._setMode(null, null)),
        this._button(this._t("remove"), () => this._remove(memory), "danger"),
      ),
    );
  }

  _remove(memory) {
    this._guard(async () => {
      await this._call("delete_memory", {
        user: this._user,
        memory_id: memory.id,
      });
      await this._load();
    });
  }

  _editor(memory) {
    const save = () =>
      this._guard(async () => {
        const text = content.value.trim();
        if (!text) {
          return;
        }
        await this._call("update_memory", {
          user: this._user,
          memory_id: memory.id,
          content: text,
          context: memory.context,
        });
        await this._load();
      });
    const cancel = () => this._setMode(null, null);
    const content = this._el("textarea", {
      rows: 3,
      value: memory.content,
      ariaLabel: this._t("edit"),
      onkeydown: (event) => {
        if (event.key === "Escape") {
          cancel();
        } else if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
          save();
        }
      },
    });
    requestAnimationFrame(() => content.focus());
    return this._el(
      "div",
      { className: "memory editing" },
      content,
      this._el(
        "div",
        { className: "buttons" },
        this._button(this._t("cancel"), cancel),
        this._button(this._t("save"), save, "primary"),
      ),
    );
  }
}

class HgaMemoryCardEditor extends HTMLElement {
  setConfig(config) {
    this._config = config || {};
    this._render();
  }

  set hass(hass) {
    this._hass = hass;
    if (!this._people) {
      this._people = [];
      this._loadPeople();
    }
    this._render();
  }

  async _loadPeople() {
    try {
      const data = await hgaMemoryCall(this._hass, { service: "get_memories", data: {} });
      this._people = data.people.map((p) => p.user);
    } catch {
      // Only an administrator can list people; the rest of the form still works.
    }
    this._render();
  }

  _schema() {
    const people = new Set(this._people || []);
    if (this._config.person) {
      people.add(this._config.person);
    }
    return [
      { name: "title", selector: { text: {} } },
      {
        name: "person",
        selector: {
          select: {
            mode: "dropdown",
            options: [...people].map((value) => ({
              value,
              label: hgaMemoryLabel(this._hass, value),
            })),
          },
        },
      },
      { name: "show_empty", selector: { boolean: {} } },
      { name: "show_search", selector: { boolean: {} } },
    ];
  }

  _render() {
    if (!this._hass || !this._config) {
      return;
    }
    if (!this._form) {
      this._form = document.createElement("ha-form");
      this._form.computeLabel = (field) =>
        hgaMemoryText(this._hass, `${field.name}_label`);
      this._form.computeHelper = (field) =>
        ["title", "person"].includes(field.name)
          ? hgaMemoryText(this._hass, `${field.name}_helper`)
          : undefined;
      this._form.addEventListener("value-changed", (event) =>
        this._changed(event.detail.value),
      );
      this.append(this._form);
    }
    this._form.hass = this._hass;
    this._form.schema = this._schema();
    this._form.data = { show_empty: false, show_search: true, ...this._config };
  }

  _changed(value) {
    const config = { ...value };
    // Only what differs from the defaults is kept, so the YAML stays short.
    for (const key of ["title", "person"]) {
      if (!config[key]) {
        delete config[key];
      }
    }
    if (!config.show_empty) {
      delete config.show_empty;
    }
    if (config.show_search !== false) {
      delete config.show_search;
    }
    this._config = config;
    this.dispatchEvent(
      new CustomEvent("config-changed", {
        detail: { config },
        bubbles: true,
        composed: true,
      }),
    );
  }
}

// HGA loads this file itself; a hand-added Resources entry would load it again.
if (!customElements.get("hga-memory-card")) {
  customElements.define("hga-memory-card", HgaMemoryCard);
  customElements.define("hga-memory-card-editor", HgaMemoryCardEditor);
  window.customCards = window.customCards || [];
  window.customCards.push({
    type: "hga-memory-card",
    name: "HGA Memories",
    description: "Browse, search, edit, move and delete long-term memories per person.",
  });
}
