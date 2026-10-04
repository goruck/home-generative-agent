const HGA_PROPOSALS_I18N = {
  en: {
    pipeline_title: "HGA Rule Pipeline",
    refresh: "Refresh",
    refreshing: "Refreshing...",
    refreshing_all: "Refreshing discovery and proposals...",
    sec_discovery: "Discovery Candidates",
    sec_filtered: "Filtered Discovery Candidates",
    sec_pending: "Proposal Drafts (Pending)",
    sec_history: "Proposal History",
    load_fail_discovery: "Failed to load discovery candidates.",
    load_fail_filtered: "Failed to load filtered discovery metadata.",
    load_fail_pending: "Failed to load pending drafts.",
    load_fail_history: "Failed to load proposal history.",
    refresh_failed: "Refresh failed: {msg}",
    unknown_error: "unknown error",
    no_discovery: "No discovery candidates.",
    no_filtered: "No filtered discovery candidates.",
    no_pending: "No pending proposal drafts.",
    no_history: "No proposal history.",
    candidate_id: "Candidate ID",
    type: "Type",
    unspecified: "unspecified",
    status: "Status",
    rule_id: "Rule ID",
    covered_rule: "Covered Rule",
    rule_state: "Rule State",
    reason: "Reason",
    semantic_key: "Semantic Key",
    promote: "Promote to Draft",
    promoting: "Promoting {id}...",
    promote_result: "Promote result: {r}",
    promote_failed: "Promote failed: {msg}",
    reject_candidate: "Reject Candidate",
    rejected_discovery: "Rejected discovery candidate {id}",
    approve: "Approve",
    approving: "Approving {id}...",
    approve_result: "Approve result: {r}",
    approve_failed: "Approve failed: {msg}",
    reject: "Reject",
    rejecting: "Rejecting {id}...",
    reject_result: "Reject result: {r}",
    reject_failed: "Reject failed: {msg}",
    unsupported_warn:
      "Unsupported: this proposal cannot be mapped to an existing deterministic template yet.",
    template_recorded: "Template request recorded for this candidate.",
    request_template: "Request New Template",
    template_requested: "Template Requested",
    template_marked: "Template request marked for {id}.",
    reactivate: "Reactivate",
    reactivating: "Reactivating {id}...",
    reactivate_result: "Reactivate result: {r}",
    reactivate_failed: "Reactivate failed: {msg}",
    deactivate: "Deactivate",
    deactivating: "Deactivating {id}...",
    deactivate_result: "Deactivate result: {r}",
    deactivate_failed: "Deactivate failed: {msg}",
    active: "active",
    inactive: "inactive",
    loaded:
      "Loaded {c} candidate(s), {f} filtered candidate(s), {p} pending draft(s), {h} historical draft(s) at {t}",
    dedupe_batch_duplicate: "Duplicate in this discovery batch",
    dedupe_existing_semantic_key: "Already covered by active/pending/recent rule idea",
    dedupe_existing_identity_hash:
      "Already covered by active/pending/recent rule idea (matched by identity)",
    dedupe_derived_only_paths: "Cites only derived values, no concrete entity",
    dedupe_entity_text_mismatch: "Text names an entity that its evidence does not cite",
    dedupe_cumulative_energy_sensor:
      "Cumulative energy sensor (kWh total), cannot become a rule",
    dedupe_unknown_network_key:
      "Refers to a network device or setting missing from the router data",
    dedupe_novel: "Novel candidate",
    dedupe_unknown: "Unknown",
    status_draft: "draft",
    status_approved: "approved",
    status_rejected: "rejected",
    status_unsupported: "unsupported",
    status_covered_by_existing_rule: "covered_by_existing_rule",
    status_ok: "ok",
    status_already_active: "already_active",
    status_exists: "exists",
    status_not_found: "not_found",
    status_error: "error",
  },
  cs: {
    pipeline_title: "HGA – pipeline pravidel",
    refresh: "Obnovit",
    refreshing: "Obnovuji…",
    refreshing_all: "Obnovuji kandidáty a návrhy…",
    sec_discovery: "Kandidáti ze zjišťování",
    sec_filtered: "Odfiltrovaní kandidáti",
    sec_pending: "Návrhy pravidel (čekající)",
    sec_history: "Historie návrhů",
    load_fail_discovery: "Kandidáty se nepodařilo načíst.",
    load_fail_filtered: "Metadata odfiltrovaných kandidátů se nepodařilo načíst.",
    load_fail_pending: "Čekající návrhy se nepodařilo načíst.",
    load_fail_history: "Historii návrhů se nepodařilo načíst.",
    refresh_failed: "Obnovení selhalo: {msg}",
    unknown_error: "neznámá chyba",
    no_discovery: "Žádní kandidáti.",
    no_filtered: "Žádní odfiltrovaní kandidáti.",
    no_pending: "Žádné čekající návrhy.",
    no_history: "Žádná historie návrhů.",
    candidate_id: "ID kandidáta",
    type: "Typ",
    unspecified: "neuvedeno",
    status: "Stav",
    rule_id: "ID pravidla",
    covered_rule: "Pokryto pravidlem",
    rule_state: "Stav pravidla",
    reason: "Důvod",
    semantic_key: "Sémantický klíč",
    promote: "Povýšit na návrh",
    promoting: "Povyšuji {id}…",
    promote_result: "Výsledek povýšení: {r}",
    promote_failed: "Povýšení selhalo: {msg}",
    reject_candidate: "Zamítnout kandidáta",
    rejected_discovery: "Kandidát zamítnut: {id}",
    approve: "Schválit",
    approving: "Schvaluji {id}…",
    approve_result: "Výsledek schválení: {r}",
    approve_failed: "Schválení selhalo: {msg}",
    reject: "Zamítnout",
    rejecting: "Zamítám {id}…",
    reject_result: "Výsledek zamítnutí: {r}",
    reject_failed: "Zamítnutí selhalo: {msg}",
    unsupported_warn:
      "Nepodporováno: tento návrh zatím nelze převést na existující deterministickou šablonu.",
    template_recorded: "Žádost o šablonu je pro tohoto kandidáta zaznamenána.",
    request_template: "Požádat o novou šablonu",
    template_requested: "Šablona vyžádána",
    template_marked: "Žádost o šablonu označena pro {id}.",
    reactivate: "Znovu aktivovat",
    reactivating: "Aktivuji {id}…",
    reactivate_result: "Výsledek aktivace: {r}",
    reactivate_failed: "Aktivace selhala: {msg}",
    deactivate: "Deaktivovat",
    deactivating: "Deaktivuji {id}…",
    deactivate_result: "Výsledek deaktivace: {r}",
    deactivate_failed: "Deaktivace selhala: {msg}",
    active: "aktivní",
    inactive: "neaktivní",
    loaded:
      "Načteno: kandidáti {c}, odfiltrovaní {f}, čekající návrhy {p}, historie {h} (v {t})",
    dedupe_batch_duplicate: "Duplicita v této dávce zjišťování",
    dedupe_existing_semantic_key:
      "Už pokryto aktivním, čekajícím nebo nedávným návrhem pravidla",
    dedupe_existing_identity_hash:
      "Už pokryto aktivním, čekajícím nebo nedávným návrhem pravidla (shoda podle identity)",
    dedupe_derived_only_paths:
      "Odkazuje jen na odvozené hodnoty, žádnou konkrétní entitu",
    dedupe_entity_text_mismatch:
      "Text jmenuje entitu, kterou její podklady neuvádějí",
    dedupe_cumulative_energy_sensor:
      "Kumulativní senzor energie (celkem kWh), nemůže se stát pravidlem",
    dedupe_unknown_network_key:
      "Odkazuje na síťového klienta nebo nastavení routeru, které v datech nejsou",
    dedupe_novel: "Nový kandidát",
    dedupe_unknown: "Neznámé",
    status_draft: "koncept",
    status_approved: "schváleno",
    status_rejected: "zamítnuto",
    status_unsupported: "nepodporováno",
    status_covered_by_existing_rule: "pokryto existujícím pravidlem",
    status_ok: "v pořádku",
    status_already_active: "už aktivní",
    status_exists: "už existuje",
    status_not_found: "nenalezeno",
    status_error: "chyba",
  },
};

class HgaProposalsCard extends HTMLElement {
  static RULE_REQUEST_URL =
    "https://github.com/goruck/home-generative-agent/issues/new";
  static RULE_REQUEST_TEMPLATE = "feature_rule_request.yml";
  static DISMISSED_KEY = "hga_proposals_card.dismissed_candidates";
  static TEMPLATE_REQUESTED_KEY = "hga_proposals_card.template_requested_candidates";

  constructor() {
    super();
    this._loading = false;
  }

  set hass(hass) {
    if (!this._hass) {
      this._hass = hass;
      this._init();
      return;
    }
    const previousLang = this._lang();
    this._hass = hass;
    if (this._lang() !== previousLang) {
      this._applyChromeLanguage();
      this._load();
    }
  }

  setConfig(config) {
    this._config = config || {};
  }

  getCardSize() {
    return 8;
  }

  async _init() {
    this.attachShadow({ mode: "open" });
    this.shadowRoot.innerHTML = `
      <style>
        .wrap { padding: 16px; font-family: sans-serif; }
        .card { border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin-bottom: 12px; }
        .card strong { overflow-wrap: anywhere; }
        .meta { color: #666; font-size: 12px; overflow-wrap: anywhere; }
        .warn { color: #b45309; font-size: 12px; margin-top: 6px; }
        .note { color: #0f766e; font-size: 12px; margin-top: 6px; }
        .row { display: flex; gap: 8px; margin-top: 8px; }
        .section { margin-top: 16px; }
        .section summary {
          cursor: pointer;
          font-weight: 600;
          margin-bottom: 8px;
        }
        .section summary::marker { font-size: 0.95em; }
        .section-content { margin-top: 8px; }
        button, a.btn-link { padding: 6px 10px; }
        a.btn-link {
          border: 1px solid #ddd;
          border-radius: 4px;
          text-decoration: none;
          color: inherit;
          display: inline-block;
        }
        a.btn-link.requested {
          border-color: #0f766e;
          color: #0f766e;
          font-weight: 600;
        }
      </style>
      <div class="wrap">
        <div class="row" style="justify-content: space-between; align-items: center;">
          <strong id="title">${this._t("pipeline_title")}</strong>
          <button id="refresh">${this._t("refresh")}</button>
        </div>
        <div id="status" class="meta"></div>
        <details class="section">
          <summary id="summary_discovery">${this._t("sec_discovery")}</summary>
          <div id="discovery" class="section-content"></div>
        </details>
        <details class="section">
          <summary id="summary_discovery_filtered">${this._t("sec_filtered")}</summary>
          <div id="discovery_filtered" class="section-content"></div>
        </details>
        <details class="section" open>
          <summary id="summary_proposals_pending">${this._t("sec_pending")}</summary>
          <div id="proposals_pending" class="section-content"></div>
        </details>
        <details class="section">
          <summary id="summary_proposals_history">${this._t("sec_history")}</summary>
          <div id="proposals_history" class="section-content"></div>
        </details>
      </div>
    `;
    this.shadowRoot.getElementById("refresh").addEventListener("click", () => {
      this._load();
    });
    await this._load();
  }

  async _callService(domain, service, data) {
    return this._hass.callWS({
      type: "call_service",
      domain,
      service,
      service_data: data,
      return_response: true,
    });
  }

  _getDismissedCandidateIds() {
    try {
      const raw = window.localStorage.getItem(HgaProposalsCard.DISMISSED_KEY);
      if (!raw) {
        return new Set();
      }
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) {
        return new Set();
      }
      return new Set(parsed.filter((value) => typeof value === "string"));
    } catch (_err) {
      return new Set();
    }
  }

  _saveDismissedCandidateIds(ids) {
    try {
      window.localStorage.setItem(
        HgaProposalsCard.DISMISSED_KEY,
        JSON.stringify(Array.from(ids))
      );
    } catch (_err) {
      // Ignore localStorage write failures.
    }
  }

  _dismissCandidate(candidateId) {
    const ids = this._getDismissedCandidateIds();
    ids.add(candidateId);
    this._saveDismissedCandidateIds(ids);
  }

  _getTemplateRequestedCandidateIds() {
    try {
      const raw = window.localStorage.getItem(
        HgaProposalsCard.TEMPLATE_REQUESTED_KEY
      );
      if (!raw) {
        return new Set();
      }
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) {
        return new Set();
      }
      return new Set(parsed.filter((value) => typeof value === "string"));
    } catch (_err) {
      return new Set();
    }
  }

  _saveTemplateRequestedCandidateIds(ids) {
    try {
      window.localStorage.setItem(
        HgaProposalsCard.TEMPLATE_REQUESTED_KEY,
        JSON.stringify(Array.from(ids))
      );
    } catch (_err) {
      // Ignore localStorage write failures.
    }
  }

  _markTemplateRequested(candidateId) {
    if (!candidateId) {
      return;
    }
    const ids = this._getTemplateRequestedCandidateIds();
    ids.add(candidateId);
    this._saveTemplateRequestedCandidateIds(ids);
  }

  _isTemplateRequested(candidateId) {
    if (!candidateId) {
      return false;
    }
    return this._getTemplateRequestedCandidateIds().has(candidateId);
  }

  _proposalDedupKey(record) {
    const explicitRuleId =
      record?.rule_id || this._inferRuleIdFromCandidate(record?.candidate || {});
    if (explicitRuleId) {
      return `rule:${explicitRuleId}`;
    }
    const candidate = record?.candidate || {};
    const suggestedType = candidate?.suggested_type || "unknown";
    const evidencePaths = Array.isArray(candidate?.evidence_paths)
      ? [...candidate.evidence_paths].sort()
      : [];
    return `candidate:${suggestedType}:${JSON.stringify(evidencePaths)}`;
  }

  _extractEntityIds(evidencePaths) {
    if (!Array.isArray(evidencePaths)) {
      return [];
    }
    const entityIds = [];
    const markers = ["entities[entity_id=", "entities[entity_ids contains "];
    // Bare-bracket format: entities[sensor.foo].state (issue #522). The
    // bracket token must be a domain-qualified entity ID, so index-based
    // brackets (entities[31].state, issue #518) still resolve nothing.
    // Mirrors _extract_entity_id_from_evidence_path in proposal_templates.py.
    const bareBracketDomains = [
      "alarm_control_panel",
      "binary_sensor",
      "camera",
      "cover",
      "input_boolean",
      "input_number",
      "light",
      "lock",
      "media_player",
      "person",
      "sensor",
      "switch",
      "vacuum",
    ];
    for (const path of evidencePaths) {
      if (typeof path !== "string") {
        continue;
      }
      let matched = false;
      for (const marker of markers) {
        const start = path.indexOf(marker);
        if (start === -1) {
          continue;
        }
        const rest = path.slice(start + marker.length);
        const end = rest.indexOf("]");
        if (end === -1) {
          continue;
        }
        // Entity IDs may be quote-wrapped (LLM output variance).
        const token = rest.slice(0, end).replace(/^['"`]+|['"`]+$/g, "");
        if (token) {
          entityIds.push(token);
        }
        matched = true;
        break;
      }
      if (!matched && path.startsWith("entities[")) {
        const rest = path.slice("entities[".length);
        const end = rest.indexOf("]");
        if (end === -1) {
          continue;
        }
        const token = rest.slice(0, end).replace(/^['"`]+|['"`]+$/g, "");
        // Prefix match tolerates an attribute suffix inside the bracket
        // (entities[sensor.x.state], LLM variance) — mirrors the server's
        // _DOT_NOTATION_ENTITY_PATTERN prefix semantics.
        const dotMatch = /^([a-z_]+\.[a-z0-9_]+)(?:[.[]|$)/.exec(token);
        if (dotMatch && bareBracketDomains.includes(dotMatch[1].split(".")[0])) {
          entityIds.push(dotMatch[1]);
        }
      }
    }
    return entityIds;
  }

  _inferRuleIdFromCandidate(candidate) {
    const text = [
      String(candidate?.suggested_type || ""),
      String(candidate?.title || ""),
      String(candidate?.summary || ""),
      String(candidate?.pattern || ""),
    ]
      .join(" ")
      .toLowerCase();
    const evidencePaths = candidate?.evidence_paths || [];
    const canonicalPaths = this._canonicalEvidencePaths(candidate);
    const entityIds = this._extractEntityIds(evidencePaths);
    // Word-bounded so "indoor"/"outdoor"/"doorbell" don't read as entries;
    // mirrors _ENTRY_TEXT_PATTERN in proposal_templates.py.
    const entryTextRe = /\b(?:doors?|windows?|entry|entries)\b/;
    const nonEntryIdTokens = [
      "motion",
      "vmd",
      "battery",
      "occupancy",
      "presence",
      "smoke",
      "gas",
      "leak",
      "moisture",
      "flood",
      "tamper",
      "vibration",
      "carbon",
      "safety",
    ];
    let entryIds = entityIds.filter(
      (entityId) =>
        // Domain restriction mirrors _find_entry_entity_ids server-side
        // (binary_sensor/cover/domainless only): without it a
        // sensor.front_door_lock_battery reads as an entry and previews
        // open_entry_* for a battery candidate (issue #522 Codex review).
        (entityId.startsWith("binary_sensor.") ||
          entityId.startsWith("cover.") ||
          !entityId.includes(".")) &&
        (entityId.includes("window") ||
          entityId.includes("door") ||
          entityId.includes("entry")) &&
        // Motion-named IDs with an entry substring (outdoor_motion,
        // doorbell_motion) are motion sensors, not entries — mirrors the
        // server's non_motion_entry_ids guard (issue #516).
        !(entityId.includes("motion") || entityId.includes("vmd"))
    );
    let entryIdsFromText = false;
    if (entryIds.length === 0 && entryTextRe.test(text)) {
      // Locale-named entity IDs (e.g. Czech "okno") carry no English entry
      // token — when the candidate text names an entry, promote unclassified
      // binary_sensor/cover IDs (mirrors proposal_templates.py, issue #504).
      entryIds = entityIds.filter(
        (entityId) =>
          (entityId.startsWith("binary_sensor.") || entityId.startsWith("cover.")) &&
          !nonEntryIdTokens.some((token) => entityId.includes(token))
      );
      entryIdsFromText = entryIds.length > 0;
    }
    // Low-battery context (issue #522) — computed before the entry legs so
    // the lock-battery precedence check below can mirror the normalizer,
    // whose lock-battery branch precedes the open-entry branches.
    const { hasLowBatterySignal } = this._lowBatteryContext(candidate, text);
    let batterySensorIds = entityIds.filter(
      (entityId) =>
        entityId.startsWith("sensor.") && entityId.includes("battery")
    );
    if (batterySensorIds.length === 0 && hasLowBatterySignal) {
      // Locale-named sensor IDs (Czech "baterie") carry no English battery
      // token: mirror _find_text_battery_sensor_entity_ids — promote a
      // SINGLE unambiguous non-excluded sensor.* evidence ID.
      const nonBatteryIdTokens = [
        "power",
        "energy",
        "watt",
        "voltage",
        "current",
        "temperature",
        "humidity",
        "illuminance",
        "lux",
        "pressure",
        "co2",
        "motion",
        "door",
        "window",
        "occupancy",
        "presence",
        "smoke",
        "gas",
        "leak",
        "moisture",
        "flood",
        "signal",
        "rssi",
        "linkquality",
      ];
      const fallback = [
        ...new Set(
          entityIds.filter(
            (entityId) =>
              entityId.startsWith("sensor.") &&
              !nonBatteryIdTokens.some((token) => entityId.includes(token))
          )
        ),
      ];
      if (fallback.length === 1) {
        batterySensorIds = fallback;
      }
    }
    // Lock-battery precedence: the normalizer routes lock + low-battery
    // candidates to low_battery_sensors (or refuses them) BEFORE any
    // open-entry branch — without this a lock+entry+battery compound would
    // preview as open_entry_* while the server registers a battery rule.
    if (
      hasLowBatterySignal &&
      entityIds.some((entityId) => entityId.startsWith("lock."))
    ) {
      return batterySensorIds.length > 0
        ? this._sanitizeRuleName(candidate?.candidate_id) || "low_battery_sensors"
        : null;
    }
    const hasNight = this._nightSignal(canonicalPaths, text);
    // Occupancy resolves through the shared evidence-first signal — mirrors
    // evidence_paths.presence_signal via _presenceSignal (issue #524). A
    // single direction value preserves the old isAway-before-isHome branch
    // order below.
    const presence = this._presenceSignal(canonicalPaths, text);
    const isAway = presence === "away";
    const isHome = presence === "home";
    if (entryIds.length > 0) {
      // Text-derived kind only for text-derived entry IDs — mirrors the
      // registry-stability gating in proposal_templates.py. Still
      // English-text-driven: there is no structured derived.* source for
      // window-vs-door kind, so locale-named IDs with non-English prose
      // degrade to the generic "entry" kind (issue #524 scope).
      const entryKind = entryIds.some((entityId) => entityId.includes("window"))
        ? "window"
        : entryIds.some((entityId) => entityId.includes("door"))
        ? "door"
        : entryIdsFromText && /\bwindows?\b/.test(text)
        ? "window"
        : entryIdsFromText && /\bdoors?\b/.test(text)
        ? "door"
        : "entry";
      if (hasNight && isAway) {
        return `open_entry_at_night_while_away_${entryKind}`;
      }
      if (hasNight && isHome) {
        return `open_entry_at_night_when_home_${entryKind}`;
      }
      // Alarm candidates fall through to the alarm_disarmed_open_entry
      // inference below, mirroring the normalizer's branch order.
      const hasAlarmSignal =
        text.includes("alarm") ||
        text.includes("disarmed") ||
        entityIds.some((entityId) => entityId.startsWith("alarm_control_panel."));
      if (hasNight && !hasAlarmSignal) {
        return `open_entry_at_night_${entryKind}`;
      }
      if (isAway) {
        return `open_entry_while_away_${entryKind}`;
      }
      if (isHome) {
        return `open_entry_when_home_${entryKind}`;
      }
    }

    // Non-lock low-battery candidates (issue #522): the lock-precedence
    // check above handled lock evidence; here the conjunctive signal plus a
    // battery sensor routes ahead of the unlocked-lock text leg.
    if (hasLowBatterySignal && batterySensorIds.length > 0) {
      return this._sanitizeRuleName(candidate?.candidate_id) || "low_battery_sensors";
    }

    if (text.includes("lock") || text.includes("unlocked")) {
      const lockId = entityIds.find((entityId) => entityId.startsWith("lock."));
      if (lockId) {
        return `unlocked_lock_when_home_${lockId.replaceAll(".", "_")}`;
      }
    }

    if (text.includes("alarm") || text.includes("disarmed")) {
      const alarmId = entityIds.find((entityId) =>
        entityId.startsWith("alarm_control_panel.")
      );
      if (alarmId && entryIds.length > 0) {
        return `alarm_disarmed_open_entry_${alarmId.replaceAll(".", "_")}`;
      }
    }

    // Motion sensors named in the candidate. Evidence-path IDs first; when
    // no MOTION-named evidence ID resolves (index-based paths like
    // entities[31].state carry no entity ID, issue #518), fall back to
    // prose IDs — the per-class gate mirrors `if not motion_ids:` in
    // proposal_templates.py, so a candidate with resolvable non-motion
    // evidence (person.*) plus an index-based motion path still predicts
    // the motion route the server takes (issue #518 review).
    const evidenceMotionIds = entityIds.filter(
      (entityId) => entityId.includes("motion") || entityId.includes("vmd")
    );
    let motionSensorIds = evidenceMotionIds;
    if (evidenceMotionIds.length === 0) {
      // Capture-group form instead of a lookbehind: a lookbehind regex
      // literal is an early SyntaxError on WebKit < 16.4 and would stop
      // the whole card script from parsing (issue #518 red-team review).
      // Mirrors _find_text_motion_entity_ids (binary_sensor-only).
      motionSensorIds = [];
      const proseIdRe = /(?:^|[^a-z0-9_.])(binary_sensor\.[a-z0-9_]+)/g;
      let proseMatch;
      while ((proseMatch = proseIdRe.exec(text)) !== null) {
        const proseId = proseMatch[1];
        if (proseId.includes("motion") || proseId.includes("vmd")) {
          motionSensorIds.push(proseId);
        }
      }
    }
    const awayMotionIds = motionSensorIds.filter((entityId) =>
      entityId.startsWith("binary_sensor.")
    );
    // Mirrors _is_away_motion_candidate in proposal_templates.py: entity
    // guards keep alarm/lock/battery candidates on their existing routing;
    // the text guards keep prose-only battery/lock/staleness/open-entry
    // predicates from silently collapsing into a plain motion rule
    // (issue #518 Codex adversarial review).
    const motionGuardsPass =
      (text.includes("motion") || text.includes("vmd")) &&
      awayMotionIds.length > 0 &&
      !/\b(?:alarms?|(?:dis)?armed)\b/.test(text) &&
      !text.includes("unavailable") &&
      !text.includes("offline") &&
      !text.includes("unreachable") &&
      // Mirrors _is_away_motion_candidate's slug-aware battery guard
      // (issue #522): a candidate whose slug says low_battery must not
      // collapse into a plain motion rule.
      !hasLowBatterySignal &&
      !/\b(?:un)?lock(?:s|ed)?\b/.test(text) &&
      !["stale", "not updated", "last seen", "last updated"].some((term) =>
        text.includes(term)
      ) &&
      !(entryTextRe.test(text) && text.includes("open")) &&
      !entityIds.some(
        (entityId) =>
          entityId.startsWith("alarm_control_panel.") ||
          entityId.startsWith("lock.") ||
          entityId.includes("battery")
      );
    // Contrastive any-hour phrasing suppresses the night gate — mirrors
    // _ANY_HOUR_TEXT_PATTERN (issue #518 red-team review).
    const anyHourRe =
      /\b(?:day (?:or|and) night|night (?:or|and) day|any ?time|any hour|24\/7|around the clock|regardless of (?:the )?time|not (?:just|only) (?:at )?night|including night(?:time)?)\b/;
    // Motion at night while away (issue #516) — mirrors the normalizer's
    // motion_detected_at_night_while_away branch.
    if (hasNight && !anyHourRe.test(text) && isAway && motionGuardsPass) {
      return "motion_detected_at_night_while_away";
    }
    // Motion while away, any hour (issue #518) — the night branch above
    // wins for night-worded candidates, mirroring the normalizer's branch
    // order. Unknown-person and camera-evidence candidates keep their
    // camera-template routing (server-side day-branch guards).
    const hasUnknownPerson =
      ["unknown", "unrecognized", "stranger", "unidentified", "indeterminate"].some(
        (term) => text.includes(term)
      ) &&
      ["person", "people", "face", "occupant", "resident"].some((term) =>
        text.includes(term)
      );
    const hasCameraEvidence =
      entityIds.some((entityId) => entityId.startsWith("camera.")) ||
      (Array.isArray(evidencePaths) &&
        evidencePaths.some(
          (path) =>
            typeof path === "string" && path.startsWith("camera_activity[")
        ));
    if (isAway && motionGuardsPass && !hasUnknownPerson && !hasCameraEvidence) {
      return "motion_detected_while_away";
    }

    if (text.includes("motion") || text.includes("camera")) {
      const cameraPath = (Array.isArray(evidencePaths) ? evidencePaths : []).find(
        (path) =>
          typeof path === "string" &&
          (path.startsWith("camera_activity[camera_entity_id=") ||
            path.startsWith("camera_activity[entity_id="))
      );
      if (!cameraPath) {
        return null;
      }
      const cameraMarker = cameraPath.includes("camera_entity_id=")
        ? "camera_activity[camera_entity_id="
        : "camera_activity[entity_id=";
      const rest = cameraPath.slice(cameraMarker.length);
      const end = rest.indexOf("]");
      if (end === -1) {
        return null;
      }
      const cameraId = rest.slice(0, end);
      return `motion_without_camera_${cameraId.replaceAll(".", "_")}`;
    }

    // Late generic battery leg — mirrors the normalizer's post-camera
    // low_battery branch: prose keeps its legacy any-of predicate, the
    // candidate_id slug counts only via the conjunctive signal (the
    // conjunctive leg above mirrors the lock-precedence and
    // fallback-promotion branches).
    if (
      batterySensorIds.length > 0 &&
      (text.includes("battery") ||
        text.includes("low") ||
        text.includes("below") ||
        hasLowBatterySignal)
    ) {
      return (
        this._sanitizeRuleName(candidate?.candidate_id) || "low_battery_sensors"
      );
    }

    return null;
  }

  _canonicalEvidencePaths(candidate) {
    // Mirrors sentinel/evidence_paths.canonicalize_evidence_path (issue
    // #524): lowercase, collapse whitespace, rewrite "!x" to "not x", and
    // fold a trailing boolean comparison into the negation ("x == false"
    // negates, "x == true" is the bare path, double negation resolves
    // positive). Consumers were literal string membership checks, so any
    // spelling variant silently fell through to the English-prose fallback.
    return (Array.isArray(candidate?.evidence_paths)
      ? candidate.evidence_paths
      : []
    )
      .filter((path) => typeof path === "string")
      .map((path) => {
        // Whitespace class is the union of JS and Python \s (U+FEFF plus
        // U+0085/U+001C-U+001F) so both mirrors canonicalize identically.
        let canonical = path
          .toLowerCase()
          .replace(/[\s\u0085\u001c-\u001f]+/g, " ")
          .trim();
        // LLMs sometimes quote the whole path (the discovery prompt itself
        // renders 'not derived.anyone_home' in quotes) — strip wrapping
        // quote characters before prefix/suffix handling.
        canonical = canonical.replace(/^['"`]+|['"`]+$/g, "").trim();
        let negated = false;
        // Loop so stacked prefixes fold by parity ("!!x", "not not x");
        // re-strip quotes each round — inner-quoted variants
        // ("not 'derived.x'") keep a wrapping quote after the prefix.
        let prefix;
        while ((prefix = canonical.match(/^(?:not\s+|!\s*)/)) !== null) {
          negated = !negated;
          canonical = canonical
            .slice(prefix[0].length)
            .replace(/^['"`]+|['"`]+$/g, "")
            .trim();
        }
        // Boolean vocabulary includes the HA state idiom (off/no, on/yes)
        // and the "is" spelling alongside JSON booleans — mirrors
        // evidence_paths.py.
        const falseSuffix = /\s*(?:==?\s*|\bis\s+)['"`]?(?:false|0|off|no)['"`]?$/;
        const trueSuffix = /\s*(?:==?\s*|\bis\s+)['"`]?(?:true|1|on|yes)['"`]?$/;
        if (falseSuffix.test(canonical)) {
          negated = !negated;
          canonical = canonical.replace(falseSuffix, "");
        } else {
          canonical = canonical.replace(trueSuffix, "");
        }
        canonical = canonical.replace(/^['"`]+|['"`]+$/g, "").trim();
        // Bare derived-key spellings alias to the canonical paths —
        // mirrors _DERIVED_ALIASES in evidence_paths.py.
        if (canonical === "anyone_home") {
          canonical = "derived.anyone_home";
        } else if (canonical === "is_night") {
          canonical = "derived.is_night";
        }
        return negated ? `not ${canonical}` : canonical;
      });
  }

  _nightSignal(canonicalPaths, text) {
    // Mirrors sentinel/evidence_paths.night_signal (issue #524): structured
    // derived.is_night evidence first; an explicit negated path blocks the
    // "night" substring fallback (which also covers "nighttime"/
    // "overnight").
    if (canonicalPaths.includes("derived.is_night")) {
      return true;
    }
    if (canonicalPaths.includes("not derived.is_night")) {
      return false;
    }
    return text.includes("night");
  }

  _presenceSignal(canonicalPaths, text) {
    // Mirrors sentinel/evidence_paths.presence_signal (issue #524):
    // structured negation, then anyone_home boolean expressions, then the
    // legacy English terms, then the bare positive path. The bare positive
    // path ranks BELOW the away terms — the LLM historically cites
    // derived.anyone_home while the prose says "while nobody is home", and
    // evidence-first there would invert such candidates to home rules.
    if (canonicalPaths.includes("not derived.anyone_home")) {
      return "away";
    }
    if (
      /anyone_home\s*(?:==?\s*|\bis\s+)(?:false|0)\b/i.test(text) ||
      /(?:\bnot\s+|!)\s*derived\.anyone_home/.test(text)
    ) {
      return "away";
    }
    if (/anyone_home\s*(?:==?\s*|\bis\s+)(?:true|1)\b/i.test(text)) {
      return "home";
    }
    // Word-bounded to mirror AWAY_TERMS_PATTERN/HOME_TERMS_PATTERN in
    // sentinel/evidence_paths.py — "present" must not match "presence" and
    // "home" must not match "anyone_home"/"armed_home" (issue #514).
    if (
      /\b(?:away|no(?:body|\s+one)\s+(?:is\s+)?(?:at\s+)?home|empty|unoccupied|no occupants|without occupants)\b/.test(
        text
      )
    ) {
      return "away";
    }
    if (
      /\b(?:someone home|occupied|home|present|occupants|residents)\b/.test(
        text
      )
    ) {
      return "home";
    }
    if (canonicalPaths.includes("derived.anyone_home")) {
      return "home";
    }
    return "any";
  }

  _lowBatteryContext(candidate, text) {
    // candidate_id is often the only English surface when the discovery LLM
    // writes prose in the home's locale — mirrors battery_text in
    // proposal_templates.py (issue #522), scoped to battery checks only so
    // slug tokens never widen night/occupancy signals. Shared by rule-id
    // inference and severity so the predicate cannot drift between them.
    // Qualifier list mirrors _LOW_BATTERY_QUALIFIERS server-side — an
    // asymmetric list breaks candidate/rule dedup (issue #522 review).
    // Prose keeps substring matching; the candidate_id slug is matched on
    // whole tokens per surface, so "backup_battery_water_flow" cannot
    // qualify via the "low" inside "flow" (server mirror).
    const qualifiers = ["low", "below", "under", "weak"];
    const slugTokens = new Set(
      String(candidate?.candidate_id || "")
        .toLowerCase()
        .split(/[^a-z0-9]+/)
    );
    const hasLowBatterySignal =
      (text.includes("battery") &&
        qualifiers.some((qualifier) => text.includes(qualifier))) ||
      (slugTokens.has("battery") &&
        qualifiers.some((qualifier) => slugTokens.has(qualifier)));
    return { hasLowBatterySignal };
  }

  _sanitizeRuleName(value) {
    return String(value || "")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "")
      .slice(0, 80);
  }

  _severityForCandidate(candidate) {
    const text = [
      String(candidate?.title || ""),
      String(candidate?.summary || ""),
      String(candidate?.pattern || ""),
      String(candidate?.suggested_type || ""),
    ]
      .join(" ")
      .toLowerCase();
    const canonicalPaths = this._canonicalEvidencePaths(candidate);
    const hasNight = this._nightSignal(canonicalPaths, text);
    // Evidence-first occupancy (issue #524): the old text-only substring
    // check never fired for structured away candidates, understating their
    // severity. No isHome — no severity branch consumes presence "home".
    const isAway = this._presenceSignal(canonicalPaths, text) === "away";
    // Low-battery candidates register at severity "low" server-side
    // regardless of night/away wording (issue #522) — checked before the
    // night/away branch so it doesn't inflate the issue-prefill severity.
    // Covers both server battery routes: the conjunctive signal and the
    // legacy prose any-of over battery-named evidence sensors. Compound
    // open-entry candidates keep their entry severity: the normalizer's
    // entry branches outrank the generic battery branch, so a "door left
    // open, battery low" candidate is an entry proposal (issue #522
    // red-team review).
    const severityBatteryEvidence = this._extractEntityIds(
      Array.isArray(candidate?.evidence_paths) ? candidate.evidence_paths : []
    ).some(
      (entityId) =>
        entityId.startsWith("sensor.") && entityId.includes("battery")
    );
    if (
      (this._lowBatteryContext(candidate, text).hasLowBatterySignal ||
        (severityBatteryEvidence &&
          (text.includes("battery") ||
            text.includes("low") ||
            text.includes("below")))) &&
      !(
        /\b(?:doors?|windows?|entry|entries)\b/.test(text) &&
        text.includes("open")
      )
    ) {
      return "low";
    }
    if (isAway || hasNight) {
      return "high";
    }
    if (text.includes("motion") && text.includes("camera")) {
      return "low";
    }
    return "medium";
  }

  _buildRuleRequestUrl(record) {
    const candidate = record?.candidate || {};
    const evidencePaths = Array.isArray(candidate?.evidence_paths)
      ? candidate.evidence_paths.filter((path) => typeof path === "string")
      : [];
    const candidateTitle = String(candidate?.title || "").trim();
    const candidateSummary = String(candidate?.summary || "").trim();
    const candidatePattern = String(candidate?.pattern || "").trim();
    const displayTitle =
      candidateTitle || String(record?.candidate_id || "New deterministic rule");
    const inferredRuleId =
      String(record?.rule_id || "").trim() || this._inferRuleIdFromCandidate(candidate);
    const ruleName = this._sanitizeRuleName(inferredRuleId || displayTitle) || "new_rule";
    const confidenceHint = Number(candidate?.confidence_hint);
    const confidenceValue = Number.isFinite(confidenceHint)
      ? String(Math.max(0, Math.min(1, confidenceHint)))
      : "0.6";
    const suggestedActions = Array.isArray(candidate?.suggested_actions)
      ? candidate.suggested_actions.filter((item) => typeof item === "string")
      : [];

    const params = new URLSearchParams();
    params.set("template", HgaProposalsCard.RULE_REQUEST_TEMPLATE);
    params.set("title", `[Rule] ${displayTitle}`);
    params.set("rule_name", ruleName);
    params.set(
      "summary",
      candidateSummary ||
        `Detected candidate ${String(record?.candidate_id || "").trim()} appears useful but is currently unsupported.`
    );
    params.set(
      "motivation",
      [
        "This proposal is currently marked unsupported in HGA and needs a deterministic template.",
        `Candidate ID: ${String(record?.candidate_id || "unknown")}`,
        candidatePattern ? `Observed pattern: ${candidatePattern}` : "",
      ]
        .filter(Boolean)
        .join("\n")
    );
    params.set(
      "snapshot_inputs",
      evidencePaths.length
        ? evidencePaths.map((path) => `- ${path}`).join("\n")
        : "- No explicit evidence_paths were included; infer from candidate description."
    );
    params.set(
      "detection_logic",
      [
        "1) Evaluate only snapshot fields listed in required evidence.",
        "2) Trigger when the candidate condition is true.",
        "3) Return no findings when any required condition is missing.",
      ].join("\n")
    );
    params.set(
      "evidence_mapping",
      evidencePaths.length
        ? evidencePaths
            .map((path, index) => `- evidence.path_${index + 1} <- ${path}`)
            .join("\n")
        : "- Map evidence fields to concrete snapshot paths used by the rule."
    );
    params.set("severity", this._severityForCandidate(candidate));
    params.set("confidence", confidenceValue);
    params.set(
      "suggested_actions",
      suggestedActions.length
        ? suggestedActions.map((action) => `- ${action}`).join("\n")
        : "- close_entry"
    );
    params.set("suppression", "Use default per-type cooldown (30 min).");
    params.set(
      "tests",
      [
        `- Trigger: ${candidateSummary || candidatePattern || "candidate condition is present in snapshot."}`,
        "- Non-trigger: same snapshot context but with condition absent.",
      ].join("\n")
    );

    return `${HgaProposalsCard.RULE_REQUEST_URL}?${params.toString()}`;
  }

  async _load() {
    if (this._loading) {
      return;
    }
    this._loading = true;
    const discovery = this.shadowRoot.getElementById("discovery");
    const discoveryFiltered = this.shadowRoot.getElementById("discovery_filtered");
    const proposalsPending = this.shadowRoot.getElementById("proposals_pending");
    const proposalsHistory = this.shadowRoot.getElementById("proposals_history");
    const status = this.shadowRoot.getElementById("status");
    const refresh = this.shadowRoot.getElementById("refresh");
    refresh.disabled = true;
    refresh.textContent = this._t("refreshing");
    discovery.innerHTML = "";
    discoveryFiltered.innerHTML = "";
    proposalsPending.innerHTML = "";
    proposalsHistory.innerHTML = "";
    status.textContent = this._t("refreshing_all");
    let discoveryResult;
    let proposalResult;
    let dynamicRuleResult;
    try {
      discoveryResult = await this._callService(
        "home_generative_agent",
        "get_discovery_records",
        { limit: 20 }
      );
      proposalResult = await this._callService(
        "home_generative_agent",
        "get_proposal_drafts",
        { limit: 50 }
      );
      dynamicRuleResult = await this._callService(
        "home_generative_agent",
        "get_dynamic_rules",
        { limit: 500 }
      );
    } catch (err) {
      discovery.innerHTML = `<div class="meta">${this._t("load_fail_discovery")}</div>`;
      discoveryFiltered.innerHTML = `<div class="meta">${this._t("load_fail_filtered")}</div>`;
      proposalsPending.innerHTML = `<div class="meta">${this._t("load_fail_pending")}</div>`;
      proposalsHistory.innerHTML = `<div class="meta">${this._t("load_fail_history")}</div>`;
      status.textContent = this._t("refresh_failed", {
        msg: err?.message || this._t("unknown_error"),
      });
      return;
    } finally {
      refresh.disabled = false;
      refresh.textContent = this._t("refresh");
      this._loading = false;
    }

    const discoveryRecords =
      (discoveryResult &&
        discoveryResult.response &&
        discoveryResult.response.records) ||
      (discoveryResult && discoveryResult.records) ||
      [];
    const proposalRecords =
      (proposalResult && proposalResult.response && proposalResult.response.records) ||
      (proposalResult && proposalResult.records) ||
      [];
    const dynamicRuleRecords =
      (dynamicRuleResult &&
        dynamicRuleResult.response &&
        dynamicRuleResult.response.records) ||
      (dynamicRuleResult && dynamicRuleResult.records) ||
      [];
    const dedupedProposals = [];
    const seenProposalIds = new Set();
    const seenProposalKeys = new Set();
    for (const proposal of proposalRecords) {
      const candidateId = proposal?.candidate_id;
      if (candidateId && seenProposalIds.has(candidateId)) {
        continue;
      }
      const proposalKey = this._proposalDedupKey(proposal);
      if (seenProposalKeys.has(proposalKey)) {
        continue;
      }
      if (candidateId) {
        seenProposalIds.add(candidateId);
      }
      seenProposalKeys.add(proposalKey);
      dedupedProposals.push(proposal);
    }

    const flattenedCandidates = [];
    const flattenedFiltered = [];
    const seen = new Set();
    const seenFiltered = new Set();
    const dismissedIds = this._getDismissedCandidateIds();
    const proposedIds = new Set(
      dedupedProposals
        .map((record) => record?.candidate_id)
        .filter((candidateId) => !!candidateId)
    );
    for (const payload of discoveryRecords) {
      const candidates = payload?.candidates || [];
      const filteredCandidates = payload?.filtered_candidates || [];
      for (const candidate of candidates) {
        if (
          !candidate?.candidate_id ||
          seen.has(candidate.candidate_id) ||
          dismissedIds.has(candidate.candidate_id) ||
          proposedIds.has(candidate.candidate_id)
        ) {
          continue;
        }
        seen.add(candidate.candidate_id);
        flattenedCandidates.push(candidate);
      }
      for (const filteredCandidate of filteredCandidates) {
        const filteredId = filteredCandidate?.candidate_id;
        if (!filteredId || seenFiltered.has(filteredId)) {
          continue;
        }
        seenFiltered.add(filteredId);
        flattenedFiltered.push(filteredCandidate);
      }
    }

    if (!flattenedCandidates.length) {
      discovery.innerHTML = `<div class="meta">${this._t("no_discovery")}</div>`;
    } else {
      for (const candidate of flattenedCandidates) {
        const card = document.createElement("div");
        card.className = "card";
        card.innerHTML = `
          <div><strong>${this._esc(candidate.title || candidate.candidate_id)}</strong></div>
          <div>${this._esc(candidate.summary || "")}</div>
          <div class="meta">${this._t("candidate_id")}: ${this._esc(candidate.candidate_id)}</div>
          <div class="meta">${this._t("type")}: ${this._esc(candidate.suggested_type || this._t("unspecified"))}</div>
        `;
        const row = document.createElement("div");
        row.className = "row";
        const promote = document.createElement("button");
        promote.textContent = this._t("promote");
        promote.addEventListener("click", async () => {
          status.textContent = this._t("promoting", { id: candidate.candidate_id });
          try {
            const response = await this._callService(
              "home_generative_agent",
              "promote_discovery_candidate",
              { candidate_id: candidate.candidate_id }
            );
            const resultStatus =
              response?.response?.status || response?.status || "ok";
            status.textContent = this._t("promote_result", { r: this._statusLabel(resultStatus) });
            if (resultStatus === "already_active" || resultStatus === "exists") {
              this._dismissCandidate(candidate.candidate_id);
            }
            await this._load();
          } catch (err) {
            status.textContent = this._t("promote_failed", {
              msg: err?.message || this._t("unknown_error"),
            });
          }
        });
        const rejectDiscovery = document.createElement("button");
        rejectDiscovery.textContent = this._t("reject_candidate");
        rejectDiscovery.addEventListener("click", async () => {
          this._dismissCandidate(candidate.candidate_id);
          status.textContent = this._t("rejected_discovery", { id: candidate.candidate_id });
          await this._load();
        });
        row.appendChild(rejectDiscovery);
        row.appendChild(promote);
        card.appendChild(row);
        discovery.appendChild(card);
      }
    }

    if (!flattenedFiltered.length) {
      discoveryFiltered.innerHTML = `<div class="meta">${this._t("no_filtered")}</div>`;
    } else {
      for (const filteredCandidate of flattenedFiltered) {
        const card = document.createElement("div");
        card.className = "card";
        card.innerHTML = `
          <div><strong>${this._esc(filteredCandidate.candidate_id)}</strong></div>
          <div class="meta">${this._t("reason")}: ${this._esc(this._dedupeReasonLabel(filteredCandidate.dedupe_reason))}</div>
          <div class="meta">${this._t("semantic_key")}: ${this._esc(filteredCandidate.semantic_key || "-")}</div>
        `;
        discoveryFiltered.appendChild(card);
      }
    }

    const pendingProposals = dedupedProposals.filter(
      (proposal) =>
        proposal?.status !== "approved" &&
        proposal?.status !== "rejected" &&
        proposal?.status !== "covered_by_existing_rule"
    );
    const historicalProposals = dedupedProposals.filter(
      (proposal) =>
        proposal?.status === "approved" ||
        proposal?.status === "rejected" ||
        proposal?.status === "covered_by_existing_rule"
    );
    const ruleStateById = new Map();
    for (const rule of dynamicRuleRecords) {
      const ruleId = String(rule?.rule_id || "");
      if (!ruleId) {
        continue;
      }
      ruleStateById.set(ruleId, Boolean(rule?.enabled ?? true));
    }
    const activeRuleIds = new Set();
    if (ruleStateById.size > 0) {
      for (const [ruleId, isEnabled] of ruleStateById.entries()) {
        if (isEnabled) {
          activeRuleIds.add(ruleId);
        }
      }
    } else {
      for (const proposal of historicalProposals) {
        const ruleId =
          proposal?.rule_id ||
          this._inferRuleIdFromCandidate(proposal?.candidate || {});
        if (ruleId) {
          activeRuleIds.add(ruleId);
        }
      }
    }
    const visiblePendingProposals = pendingProposals.filter((proposal) => {
      const effectiveRuleId =
        proposal?.rule_id ||
        this._inferRuleIdFromCandidate(proposal?.candidate || {});
      if (!effectiveRuleId) {
        return true;
      }
      return !activeRuleIds.has(effectiveRuleId);
    });

    if (!visiblePendingProposals.length) {
      proposalsPending.innerHTML = `<div class="meta">${this._t("no_pending")}</div>`;
    } else {
      for (const rec of visiblePendingProposals) {
        const candidate = rec.candidate || {};
        const isUnsupported = rec.status === "unsupported";
        const templateRequested = this._isTemplateRequested(rec.candidate_id);
        const card = document.createElement("div");
        card.className = "card";
        card.innerHTML = `
          <div><strong>${this._esc(candidate.title || rec.candidate_id)}</strong></div>
          <div class="meta">${this._t("status")}: ${this._esc(this._statusLabel(rec.status || "draft"))}</div>
          <div>${this._esc(candidate.summary || "")}</div>
          <div class="meta">${this._t("candidate_id")}: ${this._esc(rec.candidate_id)}</div>
          <div class="meta">${this._t("rule_id")}: ${this._esc(rec.rule_id || "-")}</div>
          <div class="meta">${this._t("covered_rule")}: ${this._esc(rec.covered_rule_id || "-")}</div>
          ${
            isUnsupported
              ? `<div class="warn">${this._t("unsupported_warn")}</div>`
              : ""
          }
          ${
            isUnsupported && templateRequested
              ? `<div class="note">${this._t("template_recorded")}</div>`
              : ""
          }
        `;
        const row = document.createElement("div");
        row.className = "row";
        const approve = document.createElement("button");
        approve.textContent = this._t("approve");
        approve.disabled = rec.status === "approved";
        approve.addEventListener("click", async () => {
          status.textContent = this._t("approving", { id: rec.candidate_id });
          try {
            const response = await this._callService(
              "home_generative_agent",
              "approve_rule_proposal",
              { candidate_id: rec.candidate_id }
            );
            const resultStatus =
              response?.response?.status || response?.status || "ok";
            status.textContent = this._t("approve_result", { r: this._statusLabel(resultStatus) });
            await this._load();
          } catch (err) {
            status.textContent = this._t("approve_failed", {
              msg: err?.message || this._t("unknown_error"),
            });
          }
        });
        const reject = document.createElement("button");
        reject.textContent = this._t("reject");
        reject.disabled = rec.status === "rejected";
        reject.addEventListener("click", async () => {
          status.textContent = this._t("rejecting", { id: rec.candidate_id });
          try {
            const response = await this._callService(
              "home_generative_agent",
              "reject_rule_proposal",
              { candidate_id: rec.candidate_id }
            );
            const resultStatus =
              response?.response?.status || response?.status || "ok";
            status.textContent = this._t("reject_result", { r: this._statusLabel(resultStatus) });
            await this._load();
          } catch (err) {
            status.textContent = this._t("reject_failed", {
              msg: err?.message || this._t("unknown_error"),
            });
          }
        });
        if (isUnsupported) {
          const requestLink = document.createElement("a");
          requestLink.className = `btn-link${templateRequested ? " requested" : ""}`;
          requestLink.href = this._buildRuleRequestUrl(rec);
          requestLink.target = "_blank";
          requestLink.rel = "noopener noreferrer";
          requestLink.textContent = templateRequested
            ? this._t("template_requested")
            : this._t("request_template");
          requestLink.addEventListener("click", () => {
            this._markTemplateRequested(rec.candidate_id);
            status.textContent = this._t("template_marked", { id: rec.candidate_id });
            requestLink.classList.add("requested");
            requestLink.textContent = this._t("template_requested");
            const note = card.querySelector(".note");
            if (!note) {
              const requestedNote = document.createElement("div");
              requestedNote.className = "note";
              requestedNote.textContent = this._t("template_recorded");
              card.insertBefore(requestedNote, row);
            }
          });
          row.appendChild(requestLink);
        }
        row.appendChild(approve);
        row.appendChild(reject);
        card.appendChild(row);
        proposalsPending.appendChild(card);
      }
    }

    if (!historicalProposals.length) {
      proposalsHistory.innerHTML = `<div class="meta">${this._t("no_history")}</div>`;
    } else {
      for (const rec of historicalProposals) {
        const candidate = rec.candidate || {};
        const historyRuleId = rec.rule_id || rec.covered_rule_id || null;
        const isRuleEnabled = historyRuleId
          ? ruleStateById.get(historyRuleId)
          : undefined;
        const card = document.createElement("div");
        card.className = "card";
        card.innerHTML = `
          <div><strong>${this._esc(candidate.title || rec.candidate_id)}</strong></div>
          <div class="meta">${this._t("status")}: ${this._esc(this._statusLabel(rec.status || "draft"))}</div>
          <div>${this._esc(candidate.summary || "")}</div>
          <div class="meta">${this._t("candidate_id")}: ${this._esc(rec.candidate_id)}</div>
          <div class="meta">${this._t("rule_id")}: ${this._esc(rec.rule_id || "-")}</div>
          <div class="meta">${this._t("covered_rule")}: ${this._esc(rec.covered_rule_id || "-")}</div>
          <div class="meta">${this._t("rule_state")}: ${
            historyRuleId
              ? isRuleEnabled === false
                ? this._t("inactive")
                : this._t("active")
              : "-"
          }</div>
        `;
        if (historyRuleId) {
          const row = document.createElement("div");
          row.className = "row";
          if (isRuleEnabled === false) {
            const reactivate = document.createElement("button");
            reactivate.textContent = this._t("reactivate");
            reactivate.addEventListener("click", async () => {
              status.textContent = this._t("reactivating", { id: historyRuleId });
              try {
                const response = await this._callService(
                  "home_generative_agent",
                  "reactivate_dynamic_rule",
                  { rule_id: historyRuleId }
                );
                const resultStatus =
                  response?.response?.status || response?.status || "ok";
                status.textContent = this._t("reactivate_result", { r: this._statusLabel(resultStatus) });
                await this._load();
              } catch (err) {
                status.textContent = this._t("reactivate_failed", {
                  msg: err?.message || this._t("unknown_error"),
                });
              }
            });
            row.appendChild(reactivate);
          } else {
            const deactivate = document.createElement("button");
            deactivate.textContent = this._t("deactivate");
            deactivate.addEventListener("click", async () => {
              status.textContent = this._t("deactivating", { id: historyRuleId });
              try {
                const response = await this._callService(
                  "home_generative_agent",
                  "deactivate_dynamic_rule",
                  { rule_id: historyRuleId }
                );
                const resultStatus =
                  response?.response?.status || response?.status || "ok";
                status.textContent = this._t("deactivate_result", { r: this._statusLabel(resultStatus) });
                await this._load();
              } catch (err) {
                status.textContent = this._t("deactivate_failed", {
                  msg: err?.message || this._t("unknown_error"),
                });
              }
            });
            row.appendChild(deactivate);
          }
          card.appendChild(row);
        }
        proposalsHistory.appendChild(card);
      }
    }

    status.textContent = this._t("loaded", {
      c: flattenedCandidates.length,
      f: flattenedFiltered.length,
      p: visiblePendingProposals.length,
      h: historicalProposals.length,
      t: new Date().toLocaleTimeString(),
    });
  }

  // The title, Refresh button and section titles are written once by
  // _init(); the lists re-render through _t() on every load. On a profile
  // language switch, re-label the static chrome and reload the lists so the
  // card does not sit half translated until the next Refresh.
  _applyChromeLanguage() {
    const root = this.shadowRoot;
    if (!root) {
      return;
    }
    const labels = {
      title: "pipeline_title",
      summary_discovery: "sec_discovery",
      summary_discovery_filtered: "sec_filtered",
      summary_proposals_pending: "sec_pending",
      summary_proposals_history: "sec_history",
    };
    for (const [id, key] of Object.entries(labels)) {
      const el = root.getElementById(id);
      if (el) {
        el.textContent = this._t(key);
      }
    }
    const refresh = root.getElementById("refresh");
    if (refresh) {
      refresh.textContent = this._t(refresh.disabled ? "refreshing" : "refresh");
    }
  }

  _lang() {
    const raw = this._hass?.language || this._hass?.locale?.language || "en";
    const lang = String(raw).toLowerCase().split("-")[0];
    return HGA_PROPOSALS_I18N[lang] ? lang : "en";
  }

  _t(key, params = {}) {
    const dict = HGA_PROPOSALS_I18N[this._lang()];
    let text = dict[key] ?? HGA_PROPOSALS_I18N.en[key] ?? key;
    for (const [name, value] of Object.entries(params)) {
      // Function replacer: a string replacement would expand $&, $$, $'
      // and $` found in untrusted values (candidate ids, error text).
      text = text.replaceAll(`{${name}}`, () => String(value));
    }
    return text;
  }

  // Known status codes are translated; anything else is shown verbatim.
  _statusLabel(status) {
    const key = `status_${status}`;
    return HGA_PROPOSALS_I18N.en[key] !== undefined ? this._t(key) : status;
  }

  _dedupeReasonLabel(reason) {
    const key = `dedupe_${reason}`;
    if (reason && HGA_PROPOSALS_I18N.en[key] !== undefined) {
      return this._t(key);
    }
    return reason || this._t("dedupe_unknown");
  }

  // Candidate fields originate from LLM output — escape before any innerHTML
  // interpolation so a prompt-injected candidate cannot become stored XSS.
  _esc(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

}

if (!customElements.get("hga-proposals-card")) {
  customElements.define("hga-proposals-card", HgaProposalsCard);
}

window.customCards = window.customCards || [];
window.customCards.push({
  type: "hga-proposals-card",
  name: "HGA Proposals",
  description: "Review discovery candidates, promote drafts, and approve/reject rules.",
});
