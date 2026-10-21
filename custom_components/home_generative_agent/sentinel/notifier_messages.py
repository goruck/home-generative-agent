"""
Localized fixed strings for Sentinel notification chrome.

Covers titles, subtitles, type labels, and the batch/digest/snooze
summary strings -- these are the pieces of a Sentinel notification
that are *not* LLM output: severity titles ("Security Alert"), finding-type labels
("Open entry while away"), subtitle templates, the burst-batch summary,
the daily digest, and the permanent-snooze confirmation prompt. They
are built directly in ``notifier.py`` as Python string literals and
never pass through ``strings.json`` / ``translations/*.json`` (see
discussion on #523 / #531).

This module gives them the same "pick a language, fall back to
English" treatment as ``agent/pin_messages.py``, resolved from
``hass.config.language`` -- *not* ``CONF_SENTINEL_RESPONSE_LANGUAGE``,
which is a free-text LLM prompt fragment ("Czech", "please respond in
German") rather than a language code, and so is not usable as a table
key. Using ``hass.config.language`` also means these strings translate
automatically for a Czech-configured instance even if
``sentinel_response_language`` (the LLM explainer option) is left
unset.

The deterministic per-template mobile message bodies
(``_deterministic_mobile_message`` and friends in ``notifier.py``) are
localized here too, as fixed templates with named placeholders: figures,
units, and entity names are substituted verbatim, so a translation cannot
blur them (the concern behind the #523 review discussion).

Still English-only: the pre-rendered ``summary`` of the network /
HA-security rule family. Each rule builds it at evaluation time without
access to ``hass``. ``security_device_unavailable`` is the exception: its
evidence carries the names and minutes, so the notifier re-renders the
sentence from them (``_network_summary``).

Add a new language by adding a key to ``_MESSAGES`` with the same set
of message ids as ``"en"``. Missing keys in a non-English language
fall back to the English string automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        "severity_title_high": "Security Alert",
        "severity_title_medium": "Home Alert",
        "severity_title_low": "Home Update",
        "type_open_entry_while_away": "Open entry while away",
        "type_open_entry_at_night": "Open entry at night",
        "type_open_any_window_at_night_while_away": "Window open at night",
        "type_motion_detected_at_night_while_away": "Motion at night while away",
        "type_motion_detected_while_away": "Motion while away",
        "type_unlocked_lock_at_night": "Door lock left unlocked",
        "type_camera_entry_unsecured": "Activity near unsecured entry",
        "type_alarm_disarmed_during_external_threat": (
            "Outdoor activity while alarm disarmed"
        ),
        "type_appliance_power_duration": "Appliance power duration",
        "type_low_battery_sensors": "Low battery",
        "type_entity_state_duration": "Unchanged too long",
        "type_motion_detected_at_night_while_alarm_disarmed": (
            "Motion at night, alarm disarmed"
        ),
        "type_unlabeled_rule": "Sentinel alert",
        "type_ha_sensitive_entity_exposed_without_pin": (
            "Sensitive entity exposed without PIN"
        ),
        "type_ha_new_admin_or_token": "New administrator or access token",
        "type_ha_long_lived_token_stale": "Old long-lived access token",
        "type_ha_failed_logins": "Failed login attempts",
        "type_ha_cloud_remote_ui_enabled": "Cloud remote access enabled",
        "type_ha_addon_exposed_port": "Add-on port exposed on host",
        "type_ha_addon_unprotected": "Add-on protection mode off",
        "type_ha_webhook_automation_public": "Public webhook automation",
        "type_ha_trusted_networks_bypass_login": "Trusted network login bypass",
        "type_ha_http_proxy_misconfigured": "IP banning disabled",
        "type_security_device_unavailable": "Security device unavailable",
        "type_network_unconfigured_discovered_device": "Unconfigured device discovered",
        "type_network_router_update_pending": "Router update available",
        "type_network_upnp_enabled": "UPnP enabled on router",
        "type_network_public_ip_changed": "Public IP address changed",
        "type_network_upnp_port_mapping_added": "New UPnP port mapping",
        "type_radio_new_device_joined": "New radio device joined",
        "type_network_unknown_device_joined": "New device on the network",
        "type_network_guest_client_present": "Untrusted device on the guest Wi-Fi",
        "type_network_client_usage_anomaly": "A device moved far more data than usual",
        "type_network_client_present_when": (
            "A device is on the network when it should not be"
        ),
        "type_network_client_absent_when": "A device is missing from the network",
        "type_network_posture_equals": "A router setting is at a value you watch for",
        "type_network_guest_network_idle": "Guest Wi-Fi left on unused",
        "type_network_wpa3_disabled": "WPA3 is off",
        "type_network_protection_disabled": "Router threat blocking is off",
        "type_network_ddns_enabled": "Dynamic DNS is on",
        "type_zwave_insecure_security_class": "Z-Wave device without S2 security",
        "type_zigbee_permit_join_open": "Zigbee pairing is open",
        "type_zwave_inclusion_active": "Z-Wave inclusion is active",
        "type_radio_coordinator_update_pending": "Radio coordinator update available",
        "type_alarm_state_mismatch": "Alarm state mismatch",
        "type_alarm_disarmed_open_entry": "Entry open, alarm disarmed",
        "type_baseline_deviation": "Unusual value",
        "type_time_of_day_anomaly": "Unusual for this time of day",
        "type_entity_staleness": "Data not updating",
        "type_lambda": "Custom rule alert",
        "type_motion_while_alarm_disarmed_and_home_present": (
            "Motion, alarm disarmed, someone home"
        ),
        "type_motion_without_camera_activity": "Motion without camera activity",
        "type_multiple_entries_open_count": "Several entries open",
        "type_open_entry_when_home": "Open entry while home",
        "type_sensor_threshold_condition": "Sensor above threshold",
        "type_unavailable_sensors": "Sensors unavailable",
        "type_unavailable_sensors_while_home": "Sensor unavailable while home",
        "type_unknown_person_camera_no_home": "Unknown person on camera while away",
        "type_unknown_person_camera_when_home": "Unknown person on camera while home",
        "type_unlocked_lock_when_home": "Door lock unlocked while home",
        "type_unlocked_lock_while_away": "Door lock unlocked while away",
        "subtitle_appliance_finished": "{appliance} finished",
        "subtitle_appliance_cycle_complete": "Appliance cycle complete",
        "subtitle_entry_open_alarm_disarmed": "{entry_name} open, alarm disarmed",
        "subtitle_power_deviation": "{appliance}: power {direction} than expected",
        "subtitle_reading_deviation": "{appliance}: reading {direction} than expected",
        "subtitle_entry_open": "{entity} open",
        "subtitle_state_open_for": "{entity} open for {duration}",
        "subtitle_state_unlocked_for": "{entity} unlocked for {duration}",
        "subtitle_state_unchanged_for": "{entity} unchanged for {duration}",
        "duration_hours_short": "{hours} h",
        "duration_minutes_short": "{minutes} min",
        "direction_lower": "lower",
        "direction_higher": "higher",
        "fallback_entry": "Entry",
        "fallback_sensor": "Sensor",
        "fallback_unknown_entity": "Unknown entity",
        "fallback_message": "{summary}: {entity}. {action_hint}",
        "persistent_fallback": "{summary} (severity {severity}) for {entities}. {hint}",
        "batch_title": "Home Update",
        "batch_message": "{count} home update{plural}: {type_summary}.",
        "batch_more": "\u2026and {count} more",
        "also_line": "Also: {names}",
        # Canary mode (#783): what auto-execution would have run, without acting.
        "canary_would_run": "Dry run: Sentinel would have run {actions} on {targets}.",
        "also_more": "+{count} more",
        "condition_open_while_away": "open while away",
        "condition_open_while_home": "open while home",
        "condition_open_at_night": "open at night",
        "condition_alarm_disarmed": "alarm disarmed",
        "condition_open_too_long": "open too long",
        "condition_unlocked_too_long": "unlocked too long",
        "digest_title": "Sentinel Daily Digest",
        "digest_message": (
            "Sentinel: {count} alert{plural} in the last 24 h ({sev_summary})."
        ),
        "audit_report_title": "Security audit report",
        "trust_several": (
            "Trust the ones you recognize with the Sentinel trust device service."
        ),
        "audit_report_not_run": "Checks that could not run",
        "audit_report_notes": "Notes",
        "audit_report_other": "Other",
        "audit_report_truncated": (
            "truncated; run the run_network_audit service for the rest."
        ),
        "severity_word_high": "high",
        "severity_word_medium": "medium",
        "severity_word_low": "low",
        "snooze_confirm_title": "Confirm permanent snooze",
        "snooze_confirm_message": (
            "Permanently suppress '{friendly}' alerts? "
            "This can only be undone from settings."
        ),
        # Deterministic notification bodies. These are fixed templates, so the
        # figures, units, and entity names stay exact; only the words around
        # them are localized (follow-up to #523).
        "body_a_camera": "A camera",
        "body_an_entry": "An entry",
        "body_appliance_power": (
            "{appliance} drew {power} {duration}, {threshold}. Check it."
        ),
        "body_power_watts": "about {watts} W",
        "body_power_unknown": "high power",
        "body_duration_min": "for {minutes} min",
        "body_duration_unknown": "for an extended period",
        "body_threshold_min": "above the {minutes} min threshold",
        "body_threshold_unknown": "above the configured threshold",
        "body_alarm_camera_seen_ago": (
            "{camera} saw an unrecognized person {minutes} min ago."
        ),
        "body_alarm_camera_seen": "{camera} saw an unrecognized person.",
        "body_alarm_disarmed_since": "The alarm has been disarmed since {since}.",
        "body_alarm_disarmed_now": "The alarm is currently disarmed.",
        "body_alarm_cta": "Arm the alarm or view the camera.",
        "body_entry_alarm_since": "Alarm disarmed since {since}.",
        "body_entry_alarm_now": "Alarm is disarmed.",
        "body_entry_open_alarm": (
            "{entry} is open. {alarm} Close it or snooze if expected."
        ),
        "disarm_since_other_day": "{date} at {time}",
        "body_baseline_values": (
            "{appliance}: {cur}{unit} vs usual {base}{unit}{pct}. {cta}"
        ),
        "body_baseline_pct_above": "({pct}% above normal)",
        "body_baseline_pct_below": "({pct}% below normal)",
        "body_baseline_noval": "{appliance} {noun} {direction} normal. {cta}",
        "body_baseline_noval_pct": (
            "{appliance} {noun} {pct}% {direction} normal. {cta}"
        ),
        "body_dir_above": "above",
        "body_dir_below": "below",
        "body_noun_power": "power",
        "body_noun_reading": "reading",
        "body_cta_appliance": "Check appliance.",
        "body_cta_worth": "Worth checking.",
        "body_stale_person": (
            "{name}'s location tracking has been outdated for {age}. "
            "Check if their phone is on and reachable."
        ),
        "body_stale_sensor": (
            "{name} data has been outdated for {age}. Please check the sensor."
        ),
        "age_unknown": "an extended period",
        "age_day_one": "about 1 day",
        "age_days_few": "about {count} days",
        "age_days_many": "about {count} days",
        "age_hour_one": "about 1 hour",
        "age_hours_few": "about {count} hours",
        "age_hours_many": "about {count} hours",
        "summary_device_minutes": "{name} ({minutes} min)",
        "summary_security_devices_one": (
            "{count} security device unavailable longer than {minutes} minutes: "
            "{listed}."
        ),
        "summary_security_devices_few": (
            "{count} security devices unavailable longer than {minutes} minutes: "
            "{listed}."
        ),
        "summary_security_devices_many": (
            "{count} security devices unavailable longer than {minutes} minutes: "
            "{listed}."
        ),
        "action_hint_high": "Urgent: check and secure it now.",
        "action_hint_medium": "Check soon and secure it if unexpected.",
        "action_hint_low": "Review when convenient.",
    },
    "cs": {
        "severity_title_high": "Bezpečnostní výstraha",
        "severity_title_medium": "Upozornění z domova",
        "severity_title_low": "Novinka z domova",
        "type_open_entry_while_away": "Otevřený vstup v nepřítomnosti",
        "type_open_entry_at_night": "Otevřený vstup v noci",
        "type_open_any_window_at_night_while_away": "Otevřené okno v noci",
        "type_motion_detected_at_night_while_away": ("Pohyb v noci v nepřítomnosti"),
        "type_motion_detected_while_away": "Pohyb v nepřítomnosti",
        "type_unlocked_lock_at_night": "Zámek ponechán odemčený",
        "type_camera_entry_unsecured": "Pohyb u nezajištěného vstupu",
        "type_alarm_disarmed_during_external_threat": (
            "Pohyb venku při vypnutém alarmu"
        ),
        "type_appliance_power_duration": "Doba provozu spotřebiče",
        "type_low_battery_sensors": "Slabá baterie",
        "type_entity_state_duration": "Příliš dlouho beze změny",
        "type_motion_detected_at_night_while_alarm_disarmed": (
            "Pohyb v noci při vypnutém alarmu"
        ),
        "type_unlabeled_rule": "Upozornění Sentinelu",
        "type_ha_sensitive_entity_exposed_without_pin": (
            "Citlivá entita zpřístupněna bez PIN"
        ),
        "type_ha_new_admin_or_token": "Nový administrátor nebo přístupový token",
        "type_ha_long_lived_token_stale": "Starý dlouhodobý přístupový token",
        "type_ha_failed_logins": "Neúspěšné pokusy o přihlášení",
        "type_ha_cloud_remote_ui_enabled": "Vzdálený přístup přes cloud je zapnutý",
        "type_ha_addon_exposed_port": "Port doplňku vystaven na hostiteli",
        "type_ha_addon_unprotected": "Ochranný režim doplňku vypnutý",
        "type_ha_webhook_automation_public": "Veřejná webhook automatizace",
        "type_ha_trusted_networks_bypass_login": (
            "Obejití přihlášení z důvěryhodné sítě"
        ),
        "type_ha_http_proxy_misconfigured": "Blokování IP adres vypnuté",
        "type_security_device_unavailable": "Bezpečnostní zařízení nedostupné",
        "type_network_unconfigured_discovered_device": (
            "Nalezeno nenakonfigurované zařízení"
        ),
        "type_network_router_update_pending": "K dispozici aktualizace routeru",
        "type_network_upnp_enabled": "UPnP zapnuté na routeru",
        "type_network_public_ip_changed": "Veřejná IP adresa se změnila",
        "type_network_upnp_port_mapping_added": "Nové mapování portu přes UPnP",
        "type_radio_new_device_joined": "Připojeno nové rádiové zařízení",
        "type_network_unknown_device_joined": "Nové zařízení v síti",
        "type_network_guest_client_present": "Neověřené zařízení na hostovské Wi-Fi",
        "type_network_client_usage_anomaly": (
            "Zařízení přeneslo mnohem více dat než obvykle"
        ),
        "type_network_client_present_when": "Zařízení je v síti, když by nemělo být",
        "type_network_client_absent_when": "Zařízení v síti chybí",
        "type_network_posture_equals": (
            "Nastavení routeru má hodnotu, kterou sledujete"
        ),
        "type_network_guest_network_idle": "Hostovská Wi-Fi zapnutá bez využití",
        "type_network_wpa3_disabled": "WPA3 je vypnuto",
        "type_network_protection_disabled": "Blokování hrozeb na routeru je vypnuto",
        "type_network_ddns_enabled": "Dynamické DNS je zapnuto",
        "type_zwave_insecure_security_class": "Zařízení Z-Wave bez zabezpečení S2",
        "type_zigbee_permit_join_open": "Párování Zigbee je otevřené",
        "type_zwave_inclusion_active": "Začleňování Z-Wave je aktivní",
        "type_radio_coordinator_update_pending": (
            "K dispozici aktualizace rádiového koordinátoru"
        ),
        "type_alarm_state_mismatch": "Stav alarmu neodpovídá přítomnosti",
        "type_alarm_disarmed_open_entry": "Otevřený vstup při vypnutém alarmu",
        "type_baseline_deviation": "Neobvyklá hodnota",
        "type_time_of_day_anomaly": "Neobvyklé pro tuto denní dobu",
        "type_entity_staleness": "Data se neaktualizují",
        "type_lambda": "Upozornění vlastního pravidla",
        "type_motion_while_alarm_disarmed_and_home_present": (
            "Pohyb při vypnutém alarmu, někdo je doma"
        ),
        "type_motion_without_camera_activity": "Pohyb bez aktivity kamery",
        "type_multiple_entries_open_count": "Otevřeno více vstupů",
        "type_open_entry_when_home": "Otevřený vstup v přítomnosti",
        "type_sensor_threshold_condition": "Senzor překročil práh",
        "type_unavailable_sensors": "Senzory nedostupné",
        "type_unavailable_sensors_while_home": "Senzor nedostupný v přítomnosti",
        "type_unknown_person_camera_no_home": (
            "Neznámá osoba na kameře v nepřítomnosti"
        ),
        "type_unknown_person_camera_when_home": (
            "Neznámá osoba na kameře v přítomnosti"
        ),
        "type_unlocked_lock_when_home": "Zámek odemčený v přítomnosti",
        "type_unlocked_lock_while_away": "Zámek odemčený v nepřítomnosti",
        "subtitle_appliance_finished": "{appliance} dokončeno",
        "subtitle_appliance_cycle_complete": "Cyklus spotřebiče dokončen",
        "subtitle_entry_open_alarm_disarmed": "{entry_name} otevřeno, alarm vypnutý",
        "subtitle_power_deviation": "{appliance}: odběr {direction}, než se čekalo",
        "subtitle_reading_deviation": (
            "{appliance}: hodnota {direction}, než se čekalo"
        ),
        "subtitle_entry_open": "{entity}: otevřeno",
        "subtitle_state_open_for": "{entity}: otevřeno {duration}",
        "subtitle_state_unlocked_for": "{entity}: odemčeno {duration}",
        "subtitle_state_unchanged_for": "{entity}: beze změny {duration}",
        "duration_hours_short": "{hours} h",
        "duration_minutes_short": "{minutes} min",
        "direction_lower": "nižší",
        "direction_higher": "vyšší",
        "fallback_entry": "Vstup",
        "fallback_sensor": "Senzor",
        "fallback_unknown_entity": "Neznámá entita",
        "fallback_message": "{summary}: {entity}. {action_hint}",
        "persistent_fallback": (
            "{summary} (závažnost {severity}) pro {entities}. {hint}"
        ),
        "batch_title": "Novinka z domova",
        "batch_message": "{count} novinek z domova: {type_summary}.",
        "batch_more": "\u2026a dal\u0161\u00edch {count}",
        "also_line": "Také: {names}",
        "canary_would_run": (
            "Zkušební režim: Sentinel by spustil {actions} pro {targets}."
        ),
        "also_more": "+{count} další",
        "condition_open_while_away": "otevřeno v nepřítomnosti",
        "condition_open_while_home": "otevřeno v přítomnosti",
        "condition_open_at_night": "otevřeno v noci",
        "condition_alarm_disarmed": "alarm vypnutý",
        "condition_open_too_long": "otevřeno příliš dlouho",
        "condition_unlocked_too_long": "odemčeno příliš dlouho",
        "digest_title": "Denní přehled Sentinelu",
        "digest_message": (
            "Sentinel: {count} upozornění za posledních 24 h ({sev_summary})."
        ),
        "audit_report_title": "Zpráva z bezpečnostního auditu",
        "trust_several": (
            "Ta, která znáte, označte jako důvěryhodná službou Sentinel trust device."
        ),
        "audit_report_not_run": "Kontroly, které nebylo možné provést",
        "audit_report_notes": "Poznámky",
        "audit_report_other": "Ostatní",
        "audit_report_truncated": (
            "zkráceno; zbytek získáte službou run_network_audit."
        ),
        "severity_word_high": "vysoká",
        "severity_word_medium": "střední",
        "severity_word_low": "nízká",
        "snooze_confirm_title": "Potvrdit trvalé ztlumení",
        "snooze_confirm_message": (
            "Trvale potlačit upozornění typu '{friendly}'? "
            "Lze vrátit zpět pouze v nastavení."
        ),
        "body_a_camera": "Kamera",
        "body_an_entry": "Vstup",
        "body_appliance_power": (
            "{appliance}: {power}, {duration} ({threshold}). Zkontrolujte spotřebič."
        ),
        "body_power_watts": "odběr asi {watts} W",
        "body_power_unknown": "vysoký odběr",
        "body_duration_min": "trvá {minutes} min",
        "body_duration_unknown": "trvá delší dobu",
        "body_threshold_min": "limit {minutes} min",
        "body_threshold_unknown": "přes nastavený limit",
        "body_alarm_camera_seen_ago": "{camera}: neznámá osoba před {minutes} min.",
        "body_alarm_camera_seen": "{camera}: neznámá osoba.",
        "body_alarm_disarmed_since": "Alarm je vypnutý od {since}.",
        "body_alarm_disarmed_now": "Alarm je teď vypnutý.",
        "body_alarm_cta": "Zapněte alarm nebo si prohlédněte kameru.",
        "body_entry_alarm_since": "Alarm vypnutý od {since}.",
        "body_entry_alarm_now": "Alarm je vypnutý.",
        "body_entry_open_alarm": (
            "{entry}: otevřeno. {alarm} "
            "Zavřete to, nebo ztlumte upozornění, pokud je to v pořádku."
        ),
        "disarm_since_other_day": "{date} v {time}",
        "body_baseline_values": (
            "{appliance}: {cur}{unit} oproti obvyklým {base}{unit}{pct}. {cta}"
        ),
        "body_baseline_pct_above": "({pct} % nad normálem)",
        "body_baseline_pct_below": "({pct} % pod normálem)",
        "body_baseline_noval": "{appliance}: {noun} {direction} normálem. {cta}",
        "body_baseline_noval_pct": (
            "{appliance}: {noun} {pct} % {direction} normálem. {cta}"
        ),
        "body_dir_above": "nad",
        "body_dir_below": "pod",
        "body_noun_power": "odběr",
        "body_noun_reading": "hodnota",
        "body_cta_appliance": "Zkontrolujte spotřebič.",
        "body_cta_worth": "Stojí za kontrolu.",
        "body_stale_person": (
            "Poloha osoby {name} se neaktualizuje už {age}. "
            "Zkontrolujte, zda je telefon zapnutý a dostupný."
        ),
        "body_stale_sensor": (
            "Data senzoru {name} se neaktualizují už {age}. Zkontrolujte senzor."
        ),
        "age_unknown": "delší dobu",
        "age_day_one": "asi 1 den",
        "age_days_few": "asi {count} dny",
        "age_days_many": "asi {count} dní",
        "age_hour_one": "asi 1 hodinu",
        "age_hours_few": "asi {count} hodiny",
        "age_hours_many": "asi {count} hodin",
        "summary_device_minutes": "{name} ({minutes} min)",
        "summary_security_devices_one": (
            "{count} bezpečnostní zařízení nedostupné déle než {minutes} min: "
            "{listed}."
        ),
        "summary_security_devices_few": (
            "{count} bezpečnostní zařízení nedostupná déle než {minutes} min: "
            "{listed}."
        ),
        "summary_security_devices_many": (
            "{count} bezpečnostních zařízení nedostupných déle než {minutes} min: "
            "{listed}."
        ),
        "action_hint_high": "Naléhavé: zkontrolujte a zajistěte to hned.",
        "action_hint_medium": "Zkontrolujte brzy a zajistěte, pokud je to neobvyklé.",
        "action_hint_low": "Projděte si to, až budete mít čas.",
    },
}

_DEFAULT_LANGUAGE = "en"


def _resolve_language(hass: HomeAssistant | None) -> str:
    """Resolve the message language from the HA instance's configured language."""
    if hass is None:
        return _DEFAULT_LANGUAGE
    # Guard the config attribute itself, not just language: notification
    # dispatch must never crash on language resolution, and lightweight
    # test doubles may not carry a config object at all.
    config = getattr(hass, "config", None)
    language = getattr(config, "language", None) or _DEFAULT_LANGUAGE
    base = language.split("-", 1)[0].lower()
    return base if base in _MESSAGES else _DEFAULT_LANGUAGE


def notif_msg(hass: HomeAssistant | None, key: str, /, **kwargs: Any) -> str:
    """
    Return the localized Sentinel notification-chrome string for ``key``.

    Falls back to the English string if the resolved language is
    missing the key, and to the key itself if English is missing it
    too (so a bug here shows up as an odd sentence, not a crash).
    That promise also covers formatting: this runs on the last hop of
    the notification pipeline, so a template whose placeholders drift
    from what the call site supplies degrades to the English (then
    unformatted) string instead of raising and losing the alert.
    """
    language = _resolve_language(hass)
    table = _MESSAGES.get(language, _MESSAGES[_DEFAULT_LANGUAGE])
    english = _MESSAGES[_DEFAULT_LANGUAGE].get(key)
    template = table.get(key) or english
    if template is None:
        # Unknown message id: return it verbatim and never treat it as a
        # format template — the id may be data-derived (e.g. a stored
        # severity word in the daily digest).
        return key
    if not kwargs:
        return template
    for candidate in (template, english):
        if candidate is None:
            continue
        try:
            return candidate.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            continue
    return template


def notif_language(hass: HomeAssistant | None) -> str:
    """Return the resolved notification language (``"en"`` when unsupported)."""
    return _resolve_language(hass)


def notif_plural_category(count: int) -> str:
    """
    Return ``"one"``, ``"few"`` or ``"many"`` for a whole-number *count*.

    Follows the Czech cardinal rules (1 / 2-4 / everything else, with the
    teens counting as "many"). English tables give ``few`` and ``many`` the
    same text, so the same call site serves both languages.
    """
    n = abs(int(count))
    if n == 1:
        return "one"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:  # noqa: PLR2004
        return "few"
    return "many"
