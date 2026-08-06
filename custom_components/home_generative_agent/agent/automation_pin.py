"""
Critical-action screening for LLM-authored automations.

``add_automation`` writes automation YAML and reloads Home Assistant, so an
automation whose actions reach a critical service (``lock.unlock``, say) would
otherwise install with no PIN confirmation — while the same call issued
directly through an HA tool is gated by ``_critical_action_guard``.

This module walks the *validated* automation config (the value returned by
``homeassistant.components.automation.config._async_validate_config_item``,
which has blueprints substituted and ``service:``/``action:`` normalized) and
reports the critical calls it contains.

Screening is an **allowlist over Home Assistant's own script-action taxonomy**,
not a blocklist of service names. Every step is classified with
``cv.determine_script_action``; a step is waved through only when its type is
provably inert (delay, condition, variables, …) or when it is a service call
whose domain/service/target resolve to something no critical rule matches.
Everything else is reported:

* **Device actions** (``{device_id, domain, type}``) carry no ``action:`` key at
  all, yet ``lock/device_action.py`` maps ``type: unlock`` straight to
  ``lock.unlock``. Their ``type`` is *not* reliably a service name — each
  integration maps it in its own Python, and ``cover`` turns ``set_position``
  into ``cover.set_cover_position`` — so a device action on a domain any rule
  guards is reported whenever its type does not match a rule outright.
* **State reproduction** (``scene.apply``) carries the target states inline in
  ``data``, and ``lock/reproduce_state.py`` maps the state ``unlocked`` to
  ``lock.unlock``. Screened per referenced entity. ``scene.create`` only
  snapshots current state rather than applying it, but the scene it stores can
  be activated later, so its entities are screened the same way.
* **Indirection** — activating a scene, calling a script, triggering another
  automation, firing an event — reaches actions this module cannot see without
  resolving other config. Gated rather than guessed.
* **The generic domain** (``homeassistant.turn_on`` and friends) is re-screened
  against each target entity's own domain.
* Unresolvable pieces — a templated service name, a target that is an area,
  device, label, floor, registry ID, group, or ``entity_id: all`` — are treated
  as matching any otherwise-applicable rule, and any unresolvable target of a
  generic-domain call is gated outright.
* An **unknown action type** (one Home Assistant adds after this was written)
  is gated, so the screen degrades to over-prompting rather than to a hole.

**What this cannot see.** The screen has no ``hass``, so it cannot resolve a
registry ID, expand a group or an area, or learn that a given ``switch.x`` is a
template switch whose ``turn_on`` runs a stored script. That is why an
unresolvable generic-domain target is gated rather than reasoned about. It also
matches on service names, so a *transport* service that reaches a lock without
naming the lock domain — ``mqtt.publish`` to a lock command topic is the
realistic one — is invisible to it, as is any raw protocol write that addresses
a device beneath the entity layer (``zwave_js.set_value``,
``zha.issue_zigbee_cluster_command``), in either its service or device-action
spelling. These are two separate residuals with two separate fixes: the first
needs entity-registry resolution at screen time, the second needs a rule naming
the transport service (users can add one; see ``docs/configuration.md``).
Both are tracked in TODOS.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from homeassistant.helpers import config_validation as cv

from .helpers import matches_critical_rule

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

# Depth cap for the step walk. Home Assistant's own SCRIPT_SCHEMA gives up
# well before this, so hitting it means a hand-built config: fail closed.
_MAX_STEP_DEPTH = 50

# Action types that only hold nested steps.
_CONTAINER_ACTION_TYPES = frozenset(
    {
        cv.SCRIPT_ACTION_CHOOSE,
        cv.SCRIPT_ACTION_IF,
        cv.SCRIPT_ACTION_PARALLEL,
        cv.SCRIPT_ACTION_REPEAT,
        cv.SCRIPT_ACTION_SEQUENCE,
    }
)

# Action types that cannot reach a service call.
_INERT_ACTION_TYPES = frozenset(
    {
        cv.SCRIPT_ACTION_CHECK_CONDITION,
        cv.SCRIPT_ACTION_DELAY,
        cv.SCRIPT_ACTION_SET_CONVERSATION_RESPONSE,
        cv.SCRIPT_ACTION_STOP,
        cv.SCRIPT_ACTION_VARIABLES,
        cv.SCRIPT_ACTION_WAIT_FOR_TRIGGER,
        cv.SCRIPT_ACTION_WAIT_TEMPLATE,
    }
)

# Step keys whose values hold a nested list of steps.
_SEQUENCE_KEYS = ("sequence", "then", "else", "default")

# Step keys that scope a call to something other than an explicit entity ID.
_NON_ENTITY_TARGET_KEYS = ("device_id", "area_id", "label_id", "floor_id")

# Containers within a service step that can carry targets. `data_template` is
# deprecated but still accepted by HA's SERVICE_SCHEMA and still rendered.
_TARGET_CONTAINER_KEYS = ("target", "data", "data_template", "service_data")

# Service domains that run configuration this module cannot see. `python_script`
# executes a user Python file whose sandbox permits `hass.services.call`;
# `shell_command` and `rest_command` run user-defined commands; a template
# `button`'s press field is a full script; and `conversation.process` hands
# free text to an agent that dispatches intents, so an `intent_script` whose
# action unlocks a door is reachable by sentence.
_INDIRECTION_DOMAINS = frozenset(
    {
        "automation",
        "button",
        "conversation",
        "input_button",
        "python_script",
        "rest_command",
        "script",
        "shell_command",
    }
)

# scene services that reproduce caller-supplied entity states.
_SCENE_REPRODUCE_SERVICES = frozenset({"apply", "create"})

# The generic domain forwards to each target entity's own domain.
_GENERIC_DOMAIN = "homeassistant"

# A concrete entity_id. Anything else (registry UUID, templated value) names a
# target this module cannot resolve.
_ENTITY_ID_RE = re.compile(r"^[a-z0-9_]+\.[a-z0-9_]+$")

# Entity domains that expand to *other* entities when the call runs.
_EXPANDING_DOMAINS = frozenset({"group", "scene", "script", "automation"})

# Entity domains whose targets run stored configuration of their own. Aiming a
# generic call at one of these is indirection just as much as calling
# `script.turn_on` directly, and must be gated whatever the service name is.
_INDIRECT_ENTITY_DOMAINS = frozenset(
    {"automation", "button", "input_button", "scene", "script"}
)

_ENTITY_ID_WILDCARD = "all"
_ENTITY_ID_NONE = "none"


class _TooDeepError(Exception):
    """Raised when the step walk exceeds ``_MAX_STEP_DEPTH``."""


@dataclass(frozen=True)
class CriticalAutomationCall:
    """A step inside an automation that a critical-action rule matches."""

    service: str
    entity_ids: tuple[str, ...]
    reason: str

    def describe(self) -> str:
        """Return a short, safe description of the call."""
        if self.entity_ids:
            return f"{self.service} ({', '.join(self.entity_ids)})"
        return self.service


def _unverifiable(service: str, reason: str) -> CriticalAutomationCall:
    """Build a finding for a step whose real effect cannot be determined."""
    return CriticalAutomationCall(service=service, entity_ids=(), reason=reason)


def _is_template(value: Any) -> bool:
    """Return True if a config value is (or contains) a template."""
    if isinstance(value, str):
        return "{{" in value or "{%" in value
    # Validated dynamic templates are Template objects, not strings.
    return hasattr(value, "template")


def _walk_steps(
    node: Any, depth: int = 0
) -> Iterator[tuple[Mapping[str, Any], str | None]]:
    """
    Yield every leaf step reachable from a validated actions node.

    Each item is ``(step, action_type)``; ``action_type`` is None when Home
    Assistant cannot classify the step, which callers must treat as unknown.
    Container steps are recursed into and never yielded themselves.
    """
    if depth > _MAX_STEP_DEPTH:
        raise _TooDeepError

    if isinstance(node, (list, tuple)):
        for item in node:
            yield from _walk_steps(item, depth + 1)
        return
    if not isinstance(node, dict):
        return

    try:
        action_type: str | None = cv.determine_script_action(node)
    except (ValueError, TypeError):
        action_type = None

    if action_type not in _CONTAINER_ACTION_TYPES:
        yield node, action_type
        return

    for nested in _nested_sequences(node):
        yield from _walk_steps(nested, depth + 1)


def _nested_sequences(node: Mapping[str, Any]) -> Iterator[Any]:
    """Yield each nested step list held by a container step."""
    for key in _SEQUENCE_KEYS:
        if key in node:
            yield node[key]

    choose = node.get("choose")
    if isinstance(choose, (list, tuple)):
        for option in choose:
            if isinstance(option, dict):
                yield option.get("sequence")

    repeat = node.get("repeat")
    if isinstance(repeat, dict):
        yield repeat.get("sequence")

    if "parallel" in node:
        yield node["parallel"]


def _collect_one_entity_id(value: str) -> tuple[list[str], bool]:
    """Return ([entity_id], unresolved) for a single ``entity_id`` string."""
    lowered = value.strip().lower()
    if not lowered or lowered == _ENTITY_ID_NONE:
        return [], False
    if lowered == _ENTITY_ID_WILDCARD:
        # `entity_id: all` targets every entity in the domain.
        return [], True
    if not _ENTITY_ID_RE.fullmatch(lowered):
        # Entity registry IDs are accepted in targets and resolved later, so
        # the entity this really hits is unknown at write time.
        return [], True
    if lowered.split(".", 1)[0] in _EXPANDING_DOMAINS:
        # Groups and helpers fan out to entities named nowhere in this config.
        return [], True
    return [lowered], False


def _collect_entity_ids(value: Any) -> tuple[list[str], bool]:
    """Return (entity_ids, unresolved) for an ``entity_id`` field value."""
    if value is None:
        return [], False
    if isinstance(value, (list, tuple)):
        entities: list[str] = []
        unresolved = False
        for item in value:
            item_entities, item_unresolved = _collect_entity_ids(item)
            entities.extend(item_entities)
            unresolved = unresolved or item_unresolved
        return entities, unresolved
    if not isinstance(value, str) or _is_template(value):
        # Template or unexpected type: the real targets are unknown.
        return [], True
    return _collect_one_entity_id(value)


def _targets_include_indirection(step: Mapping[str, Any]) -> bool:
    """
    Return True if any target names a script, scene, or automation entity.

    ``_collect_entity_ids`` reports these as merely unresolved, which is the
    right answer for a group but too weak here: the target runs configuration
    of its own, so it has to be gated regardless of the service name.
    """
    containers: list[Any] = [step]
    containers.extend(step[key] for key in _TARGET_CONTAINER_KEYS if key in step)

    for container in containers:
        if not isinstance(container, dict):
            continue
        value = container.get("entity_id")
        candidates = value if isinstance(value, (list, tuple)) else [value]
        for candidate in candidates:
            if not isinstance(candidate, str):
                continue
            domain, _, object_id = candidate.strip().lower().partition(".")
            if object_id and domain in _INDIRECT_ENTITY_DOMAINS:
                return True
    return False


def _step_targets(step: Mapping[str, Any]) -> tuple[list[str], bool]:
    """Return (entity_ids, unresolved_target) for one step."""
    entities: list[str] = []
    unresolved = False

    containers: list[Any] = [step]
    containers.extend(step[key] for key in _TARGET_CONTAINER_KEYS if key in step)

    for container in containers:
        if not isinstance(container, dict):
            # e.g. a fully templated `target:`.
            unresolved = True
            continue
        found, found_unresolved = _collect_entity_ids(container.get("entity_id"))
        entities.extend(found)
        unresolved = unresolved or found_unresolved
        if any(container.get(key) for key in _NON_ENTITY_TARGET_KEYS):
            unresolved = True

    # No resolvable entity at all: fail closed for entity_match rules.
    return entities, unresolved or not entities


def _service_name(step: Mapping[str, Any]) -> tuple[str, str, bool]:
    """Return (domain, service, templated) for a call_service step."""
    if "service_template" in step:
        return "", "", True

    # Validation normalizes `service:` to `action:`; accept both so the scanner
    # is still correct if it is ever handed a pre-validation config.
    raw = step.get("action", step.get("service"))
    if raw is None or _is_template(raw) or not isinstance(raw, str):
        return "", "", True

    lowered = raw.strip().lower()
    if not _ENTITY_ID_RE.fullmatch(lowered):
        # Not a `domain.service` name — e.g. a mobile-push `action:` button
        # label nested in `data`. Anything stranger than a slug pair also lands
        # here, which keeps unvalidated text out of the PIN prompt.
        return "", "", False
    domain, _, service = lowered.partition(".")
    return domain, service, False


def _domain_is_guarded(domain: str, rules: Iterable[Mapping[str, str]]) -> bool:
    """Return True if any critical rule could apply to this entity domain."""
    return any(
        not (rule.get("domain") or "") or (rule.get("domain") or "").lower() == domain
        for rule in rules
    )


def _screen_state_reproduction(
    step: Mapping[str, Any], rules: Sequence[Mapping[str, str]]
) -> list[CriticalAutomationCall]:
    """Screen `scene.apply` / `scene.create`, which set states inline."""
    unreadable = _unverifiable(
        "scene.apply",
        "the automation reproduces entity states that cannot be read here",
    )
    data = step.get("data")
    if not isinstance(data, dict):
        return [unreadable]

    candidates: list[Any] = []
    for key in ("entities", "snapshot_entities"):
        block = data.get(key)
        if isinstance(block, (dict, list, tuple)):
            # Iterating a dict yields its entity_id keys, which is what we want.
            candidates.extend(block)
        elif block is not None:
            return [unreadable]

    found: list[CriticalAutomationCall] = []
    for candidate in candidates:
        entities, _unresolved = _collect_entity_ids(candidate)
        if not entities:
            found.append(unreadable)
            continue
        # Which service the reproduction ends up calling depends on the
        # requested state, so any rule for the entity's domain has to count.
        found.extend(
            CriticalAutomationCall(
                service="scene.apply",
                entity_ids=(entity,),
                reason=(
                    "the automation sets entity states directly, which can "
                    "perform a critical action"
                ),
            )
            for entity in entities
            if _domain_is_guarded(entity.split(".", 1)[0], rules)
        )
    return found


def _screen_call_service(
    step: Mapping[str, Any], rules: Sequence[Mapping[str, str]]
) -> list[CriticalAutomationCall]:
    """Screen a single ``call_service`` step."""
    domain, service, templated = _service_name(step)
    if templated:
        return [
            _unverifiable(
                "<templated service>",
                "the automation builds a service call from a template, so the "
                "action it will run cannot be verified",
            )
        ]
    if not domain or not service:
        # Not a service call (e.g. a mobile-push `action:` label in `data`).
        return []

    special = _screen_special_domain(step, domain, service, rules)
    if special is not None:
        return special

    entities, unresolved = _step_targets(step)
    if not matches_critical_rule(
        domain=domain,
        service=service,
        entity_ids=entities,
        critical_actions=rules,
        unresolved_target=unresolved,
    ):
        return []
    return [
        CriticalAutomationCall(
            service=f"{domain}.{service}",
            entity_ids=tuple(entities),
            reason="the automation performs a critical action",
        )
    ]


def _screen_special_domain(
    step: Mapping[str, Any],
    domain: str,
    service: str,
    rules: Sequence[Mapping[str, str]],
) -> list[CriticalAutomationCall] | None:
    """
    Screen domains that do not mean what their name says.

    Returns None when the call is an ordinary service call for the caller to
    match against the rules directly.
    """
    if domain == "scene":
        if service in _SCENE_REPRODUCE_SERVICES:
            return _screen_state_reproduction(step, rules)
        # Anything else on the scene domain activates a stored scene whose
        # entity states live in another config file.
        return [
            _unverifiable(
                f"{domain}.{service}",
                "the automation runs another scene, script, or automation whose "
                "actions cannot be checked here",
            )
        ]
    if domain in _INDIRECTION_DOMAINS:
        return [
            _unverifiable(
                f"{domain}.{service}",
                "the automation runs another scene, script, or automation whose "
                "actions cannot be checked here",
            )
        ]
    if domain == _GENERIC_DOMAIN:
        if _targets_include_indirection(step):
            # `homeassistant.turn_on` aimed at `script.x` forwards to
            # `script.turn_on` and runs the script. Gating only the explicitly
            # named `script.`/`scene.` domains above would let the generic
            # alias walk straight past that boundary.
            return [
                _unverifiable(
                    f"{domain}.{service}",
                    "the automation runs another scene, script, or automation whose "
                    "actions cannot be checked here",
                )
            ]
        entities, unresolved = _step_targets(step)
        return _screen_generic_domain(service, entities, unresolved, rules)
    return None


def _screen_generic_domain(
    service: str,
    entities: Sequence[str],
    unresolved: bool,  # noqa: FBT001
    rules: Sequence[Mapping[str, str]],
) -> list[CriticalAutomationCall]:
    """Screen `homeassistant.*`, which forwards to each target's own domain."""
    found = [
        CriticalAutomationCall(
            service=f"{entity.split('.', 1)[0]}.{service}",
            entity_ids=(entity,),
            reason="the automation performs a critical action",
        )
        for entity in entities
        if matches_critical_rule(
            domain=entity.split(".", 1)[0],
            service=service,
            entity_ids=[entity],
            critical_actions=rules,
        )
    ]

    # Targets can be part concrete and part unresolvable (an area alongside a
    # named entity, say). The named ones are screened above; the rest cannot be
    # screened at all. Scoping this by service name was tried and was unsound:
    # an area, group, registry ID, or template can resolve to a script or to a
    # template entity whose `turn_on` runs a stored script, and no rule names
    # `turn_on`. Without the entity registry the only honest answer is to ask.
    if unresolved:
        found.append(
            _unverifiable(
                f"{_GENERIC_DOMAIN}.{service}",
                "the automation uses a generic service call whose targets "
                "cannot all be resolved here",
            )
        )
    return found


def _screen_device_action(
    step: Mapping[str, Any], rules: Sequence[Mapping[str, str]]
) -> list[CriticalAutomationCall]:
    """Screen a device action, which has no service key of its own."""
    domain = str(step.get("domain") or "").lower()
    action_kind = str(step.get("type") or "").lower()
    if not domain or not action_kind:
        return [
            _unverifiable(
                "<device action>",
                "the automation runs a device action that cannot be identified",
            )
        ]

    if domain in _INDIRECTION_DOMAINS:
        # Every indirection domain has a device-action spelling too: a
        # `{device_id, domain: button, type: press}` step reaches the same
        # template button whose press field is a full script as the
        # `button.press` service call does.
        return [
            _unverifiable(
                f"{domain}.{action_kind}",
                "the automation runs another scene, script, or automation whose "
                "actions cannot be checked here",
            )
        ]

    entities, unresolved = _step_targets(step)
    if matches_critical_rule(
        domain=domain,
        service=action_kind,
        entity_ids=entities,
        critical_actions=rules,
        unresolved_target=unresolved,
    ):
        return [
            CriticalAutomationCall(
                service=f"{domain}.{action_kind}",
                entity_ids=tuple(entities),
                reason="the automation performs a critical action on a device",
            )
        ]

    # A device action's `type` is not the service name. Each integration maps
    # it in its own Python (cover turns `set_position` into
    # `cover.set_cover_position`, `open` into `cover.open_cover`), and that
    # mapping cannot be introspected. Matching the raw type is therefore only
    # sound when it happens to match; when it does not, any rule on the domain
    # still has to count.
    if _domain_is_guarded(domain, rules):
        return [
            CriticalAutomationCall(
                service=f"{domain}.{action_kind}",
                entity_ids=tuple(entities),
                reason=(
                    "the automation runs a device action on a guarded domain, "
                    "and the service it maps to cannot be resolved here"
                ),
            )
        ]
    return []


def _screen_step(
    step: Mapping[str, Any],
    action_type: str | None,
    rules: Sequence[Mapping[str, str]],
) -> list[CriticalAutomationCall]:
    """Screen one classified leaf step."""
    if action_type in _INERT_ACTION_TYPES:
        return []
    if action_type == cv.SCRIPT_ACTION_CALL_SERVICE:
        return _screen_call_service(step, rules)
    if action_type == cv.SCRIPT_ACTION_DEVICE_AUTOMATION:
        return _screen_device_action(step, rules)
    if action_type == cv.SCRIPT_ACTION_ACTIVATE_SCENE:
        return [
            _unverifiable(
                "scene",
                "the automation activates a scene whose entity states cannot "
                "be checked here",
            )
        ]
    if action_type == cv.SCRIPT_ACTION_FIRE_EVENT:
        return [
            _unverifiable(
                "event",
                "the automation fires an event that other automations may act on",
            )
        ]
    # An action type this screen does not know — including one a future Home
    # Assistant release adds. Gate rather than guess.
    return [
        _unverifiable(
            str(action_type or "<unknown action>"),
            "the automation contains a step this integration cannot classify",
        )
    ]


def find_critical_automation_calls(
    validated_config: Mapping[str, Any],
    critical_actions: Iterable[Mapping[str, str]],
) -> list[CriticalAutomationCall]:
    """
    Return the critical calls an automation config would install.

    ``validated_config`` is normally the config returned by
    ``_async_validate_config_item``, so that blueprints are already substituted
    and ``service:`` keys normalized to ``action:``. Pre-validation spellings
    are also accepted as defence in depth, but blueprint substitution only
    happens on validated input.
    """
    rules: Sequence[Mapping[str, str]] = list(critical_actions)
    actions = validated_config.get("actions")
    if actions is None:
        actions = validated_config.get("action")

    found: list[CriticalAutomationCall] = []
    try:
        for step, action_type in _walk_steps(actions):
            found.extend(_screen_step(step, action_type, rules))
    except _TooDeepError:
        found.append(
            _unverifiable(
                "<deeply nested automation>",
                "the automation nests steps too deeply to check",
            )
        )
    return found
