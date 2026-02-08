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
    }
    this._hass = hass;
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
        .meta { color: #666; font-size: 12px; }
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
          <strong>HGA Rule Pipeline</strong>
          <button id="refresh">Refresh</button>
        </div>
        <div id="status" class="meta"></div>
        <details class="section">
          <summary>Discovery Candidates</summary>
          <div id="discovery" class="section-content"></div>
        </details>
        <details class="section">
          <summary>Filtered Discovery Candidates</summary>
          <div id="discovery_filtered" class="section-content"></div>
        </details>
        <details class="section" open>
          <summary>Proposal Drafts (Pending)</summary>
          <div id="proposals_pending" class="section-content"></div>
        </details>
        <details class="section">
          <summary>Proposal History</summary>
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
    for (const path of evidencePaths) {
      if (typeof path !== "string") {
        continue;
      }
      const marker = "entities[entity_id=";
      const start = path.indexOf(marker);
      if (start === -1) {
        continue;
      }
      const rest = path.slice(start + marker.length);
      const end = rest.indexOf("]");
      if (end === -1) {
        continue;
      }
      entityIds.push(rest.slice(0, end));
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
    const entityIds = this._extractEntityIds(evidencePaths);
    const entryIds = entityIds.filter(
      (entityId) =>
        entityId.includes("window") ||
        entityId.includes("door") ||
        entityId.includes("entry")
    );
    const hasNight =
      evidencePaths.includes("derived.is_night") || text.includes("night");
    const isAway =
      text.includes("away") ||
      text.includes("no one home") ||
      text.includes("nobody home") ||
      text.includes("empty") ||
      text.includes("unoccupied");
    const isHome =
      text.includes("home") ||
      text.includes("occupied") ||
      text.includes("present") ||
      evidencePaths.includes("derived.anyone_home");
    if (entryIds.length > 0) {
      const entryKind = entryIds.some((entityId) => entityId.includes("window"))
        ? "window"
        : entryIds.some((entityId) => entityId.includes("door"))
        ? "door"
        : "entry";
      if (hasNight && isAway) {
        return `open_entry_at_night_while_away_${entryKind}`;
      }
      if (hasNight && isHome) {
        return `open_entry_at_night_when_home_${entryKind}`;
      }
      if (isAway) {
        return `open_entry_while_away_${entryKind}`;
      }
      if (isHome) {
        return `open_entry_when_home_${entryKind}`;
      }
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

    return null;
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
    const hasNight =
      (Array.isArray(candidate?.evidence_paths)
        ? candidate.evidence_paths
        : []
      ).includes("derived.is_night") || text.includes("night");
    const isAway =
      text.includes("away") ||
      text.includes("no one home") ||
      text.includes("nobody home") ||
      text.includes("empty") ||
      text.includes("unoccupied");
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
    refresh.textContent = "Refreshing...";
    discovery.innerHTML = "";
    discoveryFiltered.innerHTML = "";
    proposalsPending.innerHTML = "";
    proposalsHistory.innerHTML = "";
    status.textContent = "Refreshing discovery and proposals...";
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
      discovery.innerHTML = `<div class="meta">Failed to load discovery candidates.</div>`;
      discoveryFiltered.innerHTML = `<div class="meta">Failed to load filtered discovery metadata.</div>`;
      proposalsPending.innerHTML = `<div class="meta">Failed to load pending drafts.</div>`;
      proposalsHistory.innerHTML = `<div class="meta">Failed to load proposal history.</div>`;
      status.textContent = `Refresh failed: ${err?.message || "unknown error"}`;
      return;
    } finally {
      refresh.disabled = false;
      refresh.textContent = "Refresh";
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
      discovery.innerHTML = `<div class="meta">No discovery candidates.</div>`;
    } else {
      for (const candidate of flattenedCandidates) {
        const card = document.createElement("div");
        card.className = "card";
        card.innerHTML = `
          <div><strong>${candidate.title || candidate.candidate_id}</strong></div>
          <div>${candidate.summary || ""}</div>
          <div class="meta">Candidate ID: ${candidate.candidate_id}</div>
          <div class="meta">Type: ${candidate.suggested_type || "unspecified"}</div>
        `;
        const row = document.createElement("div");
        row.className = "row";
        const promote = document.createElement("button");
        promote.textContent = "Promote to Draft";
        promote.addEventListener("click", async () => {
          status.textContent = `Promoting ${candidate.candidate_id}...`;
          try {
            const response = await this._callService(
              "home_generative_agent",
              "promote_discovery_candidate",
              { candidate_id: candidate.candidate_id }
            );
            const resultStatus =
              response?.response?.status || response?.status || "ok";
            status.textContent = `Promote result: ${resultStatus}`;
            if (resultStatus === "already_active" || resultStatus === "exists") {
              this._dismissCandidate(candidate.candidate_id);
            }
            await this._load();
          } catch (err) {
            status.textContent = `Promote failed: ${
              err?.message || "unknown error"
            }`;
          }
        });
        const rejectDiscovery = document.createElement("button");
        rejectDiscovery.textContent = "Reject Candidate";
        rejectDiscovery.addEventListener("click", async () => {
          this._dismissCandidate(candidate.candidate_id);
          status.textContent = `Rejected discovery candidate ${candidate.candidate_id}`;
          await this._load();
        });
        row.appendChild(rejectDiscovery);
        row.appendChild(promote);
        card.appendChild(row);
        discovery.appendChild(card);
      }
    }

    if (!flattenedFiltered.length) {
      discoveryFiltered.innerHTML = `<div class="meta">No filtered discovery candidates.</div>`;
    } else {
      for (const filteredCandidate of flattenedFiltered) {
        const card = document.createElement("div");
        card.className = "card";
        card.innerHTML = `
          <div><strong>${filteredCandidate.candidate_id}</strong></div>
          <div class="meta">Reason: ${this._dedupeReasonLabel(filteredCandidate.dedupe_reason)}</div>
          <div class="meta">Semantic Key: ${filteredCandidate.semantic_key || "-"}</div>
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
      proposalsPending.innerHTML = `<div class="meta">No pending proposal drafts.</div>`;
    } else {
      for (const rec of visiblePendingProposals) {
        const candidate = rec.candidate || {};
        const isUnsupported = rec.status === "unsupported";
        const templateRequested = this._isTemplateRequested(rec.candidate_id);
        const card = document.createElement("div");
        card.className = "card";
        card.innerHTML = `
          <div><strong>${candidate.title || rec.candidate_id}</strong></div>
          <div class="meta">Status: ${rec.status || "draft"}</div>
          <div>${candidate.summary || ""}</div>
          <div class="meta">Candidate ID: ${rec.candidate_id}</div>
          <div class="meta">Rule ID: ${rec.rule_id || "-"}</div>
          <div class="meta">Covered Rule: ${rec.covered_rule_id || "-"}</div>
          ${
            isUnsupported
              ? `<div class="warn">Unsupported: this proposal cannot be mapped to an existing deterministic template yet.</div>`
              : ""
          }
          ${
            isUnsupported && templateRequested
              ? `<div class="note">Template request recorded for this candidate.</div>`
              : ""
          }
        `;
        const row = document.createElement("div");
        row.className = "row";
        const approve = document.createElement("button");
        approve.textContent = "Approve";
        approve.disabled = rec.status === "approved";
        approve.addEventListener("click", async () => {
          status.textContent = `Approving ${rec.candidate_id}...`;
          try {
            const response = await this._callService(
              "home_generative_agent",
              "approve_rule_proposal",
              { candidate_id: rec.candidate_id }
            );
            const resultStatus =
              response?.response?.status || response?.status || "ok";
            status.textContent = `Approve result: ${resultStatus}`;
            await this._load();
          } catch (err) {
            status.textContent = `Approve failed: ${
              err?.message || "unknown error"
            }`;
          }
        });
        const reject = document.createElement("button");
        reject.textContent = "Reject";
        reject.disabled = rec.status === "rejected";
        reject.addEventListener("click", async () => {
          status.textContent = `Rejecting ${rec.candidate_id}...`;
          try {
            const response = await this._callService(
              "home_generative_agent",
              "reject_rule_proposal",
              { candidate_id: rec.candidate_id }
            );
            const resultStatus =
              response?.response?.status || response?.status || "ok";
            status.textContent = `Reject result: ${resultStatus}`;
            await this._load();
          } catch (err) {
            status.textContent = `Reject failed: ${
              err?.message || "unknown error"
            }`;
          }
        });
        if (isUnsupported) {
          const requestLink = document.createElement("a");
          requestLink.className = `btn-link${templateRequested ? " requested" : ""}`;
          requestLink.href = this._buildRuleRequestUrl(rec);
          requestLink.target = "_blank";
          requestLink.rel = "noopener noreferrer";
          requestLink.textContent = templateRequested
            ? "Template Requested"
            : "Request New Template";
          requestLink.addEventListener("click", () => {
            this._markTemplateRequested(rec.candidate_id);
            status.textContent = `Template request marked for ${rec.candidate_id}.`;
            requestLink.classList.add("requested");
            requestLink.textContent = "Template Requested";
            const note = card.querySelector(".note");
            if (!note) {
              const requestedNote = document.createElement("div");
              requestedNote.className = "note";
              requestedNote.textContent =
                "Template request recorded for this candidate.";
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
      proposalsHistory.innerHTML = `<div class="meta">No proposal history.</div>`;
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
          <div><strong>${candidate.title || rec.candidate_id}</strong></div>
          <div class="meta">Status: ${rec.status || "draft"}</div>
          <div>${candidate.summary || ""}</div>
          <div class="meta">Candidate ID: ${rec.candidate_id}</div>
          <div class="meta">Rule ID: ${rec.rule_id || "-"}</div>
          <div class="meta">Covered Rule: ${rec.covered_rule_id || "-"}</div>
          <div class="meta">Rule State: ${
            historyRuleId
              ? isRuleEnabled === false
                ? "inactive"
                : "active"
              : "-"
          }</div>
        `;
        if (historyRuleId) {
          const row = document.createElement("div");
          row.className = "row";
          if (isRuleEnabled === false) {
            const reactivate = document.createElement("button");
            reactivate.textContent = "Reactivate";
            reactivate.addEventListener("click", async () => {
              status.textContent = `Reactivating ${historyRuleId}...`;
              try {
                const response = await this._callService(
                  "home_generative_agent",
                  "reactivate_dynamic_rule",
                  { rule_id: historyRuleId }
                );
                const resultStatus =
                  response?.response?.status || response?.status || "ok";
                status.textContent = `Reactivate result: ${resultStatus}`;
                await this._load();
              } catch (err) {
                status.textContent = `Reactivate failed: ${
                  err?.message || "unknown error"
                }`;
              }
            });
            row.appendChild(reactivate);
          } else {
            const deactivate = document.createElement("button");
            deactivate.textContent = "Deactivate";
            deactivate.addEventListener("click", async () => {
              status.textContent = `Deactivating ${historyRuleId}...`;
              try {
                const response = await this._callService(
                  "home_generative_agent",
                  "deactivate_dynamic_rule",
                  { rule_id: historyRuleId }
                );
                const resultStatus =
                  response?.response?.status || response?.status || "ok";
                status.textContent = `Deactivate result: ${resultStatus}`;
                await this._load();
              } catch (err) {
                status.textContent = `Deactivate failed: ${
                  err?.message || "unknown error"
                }`;
              }
            });
            row.appendChild(deactivate);
          }
          card.appendChild(row);
        }
        proposalsHistory.appendChild(card);
      }
    }

    status.textContent =
      `Loaded ${flattenedCandidates.length} candidate(s), ` +
      `${flattenedFiltered.length} filtered candidate(s), ` +
      `${visiblePendingProposals.length} pending draft(s), ` +
      `${historicalProposals.length} historical draft(s) at ${new Date().toLocaleTimeString()}`;
  }

  _dedupeReasonLabel(reason) {
    const reasonMap = {
      batch_duplicate: "Duplicate in this discovery batch",
      existing_semantic_key: "Already covered by active/pending/recent rule idea",
      novel: "Novel candidate",
    };
    return reasonMap[reason] || reason || "Unknown";
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
