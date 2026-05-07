from __future__ import annotations

"""
main_study.py
============================
Pre-session dialogs for the main study:

1. ``run_runtime_parameter_dialog`` - lets the operator review / edit the
   most important session-level parameters before anything is recorded.
   Defaults come from .env (via config.py) / session.json; edited values
   are returned for persistence in protocol.json.
2. ``run_compliance_checklist`` - pre-flight boolean checklist
   (phone in airplane mode, smartwatch off, ...).
3. ``run_questionnaire_section`` - generic renderer for schema-driven
   sections (Q1 trait, Q2 state, Q3 environment, post-session). The schema
   is loaded from configs/main_study_proband_schema.json.
4. ``check_exclusion_criteria`` - evaluates the hard exclusion rules on
   the collected Q1 answers and returns ("exclude", reason) pairs.

All dialogs are thin wrappers around ``psychopy.gui.Dlg``; the module
falls back to stdin-based prompts only for headless smoke tests.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

LANG_EN = "en"
LANG_DE = "de"

TEXT_DE = {
    "Participant ID": "Teilnehmer-ID",
    "Interface language / Sprache": "Interface language / Sprache",
    "Acquisition mode": "Aufnahmemodus",
    "Trials per target/frequency": "Trials pro Ziel/Frequenz",
    "Trial duration (s)": "Trial-Dauer (s)",
    "Cue duration (s)": "Cue-Dauer (s)",
    "Stimulus distance from center (deg)": "Stimulus-Abstand vom Zentrum (Grad)",
    "Stimulus size (deg)": "Stimulus-Größe (Grad)",
    "Baseline eyes-open (s)": "Baseline mit offenen Augen (s)",
    "Include eyes-closed baseline?": "Baseline mit geschlossenen Augen aufnehmen?",
    "Baseline eyes-closed (s)": "Baseline mit geschlossenen Augen (s)",
    "Standard block break (s)": "Kurze Blockpause (s)",
    "Long break (s) - middle of study": "Lange Pause in der Mitte (s)",
    "Impedance stable hold - initial (s)": "Impedanz stabil zu Beginn (s)",
    "Impedance stable hold - between (s)": "Impedanz stabil zwischen Blöcken (s)",
    "Impedance quality threshold (0..4)": "Impedanz-Qualitätsschwelle (0..4)",
    "Idle trials per source block": "Idle-Trials pro aktivem Block",
    "Random seed base": "Basiswert für Zufallsreihenfolge",
    "Written consent is confirmed": "Schriftliche Einwilligung liegt vor",
    "Phone is in airplane mode": "Handy ist im Flugmodus",
    "Smartwatch is removed or off": "Smartwatch ist abgelegt oder ausgeschaltet",
    "Room light is stable and not flickering": "Raumlicht ist stabil und flackert nicht",
    "Screen brightness is set to the planned level": "Bildschirmhelligkeit ist wie geplant eingestellt",
    "Display refresh rate is set to 60 Hz": "Bildwiederholrate ist auf 60 Hz eingestellt",
    "Night light or blue-light filter is off": "Nachtmodus oder Blaulichtfilter ist ausgeschaltet",
    "HDR and automatic brightness are off": "HDR und automatische Helligkeit sind ausgeschaltet",
    "Viewing distance was measured and set": "Sitzabstand wurde gemessen und eingestellt",
    "Participant is seated comfortably and centered": "Teilnehmende Person sitzt bequem und mittig",
    "Participant knows they can pause or stop at any time": "Teilnehmende Person weiß, dass sie jederzeit pausieren oder abbrechen kann",
    "Participant is ready": "Teilnehmende Person ist bereit",
    "Participant info": "Teilnehmerdaten",
    "Current state": "Aktueller Zustand",
    "Room": "Raum",
    "After session": "Nach der Sitzung",
    "Age": "Alter",
    "Sex": "Geschlecht",
    "Handedness": "Händigkeit",
    "Visual aid used today": "Heute genutzte Sehhilfe",
    "Normal or corrected-to-normal vision": "Normale oder korrigierte Sehschärfe",
    "Vision problem not corrected today": "Heute nicht korrigiertes Sehproblem",
    "Known color vision problem": "Bekannte Farbsehschwäche",
    "History of epilepsy or seizures": "Epilepsie oder Krampfanfälle in der Vorgeschichte",
    "Strong reaction to flicker or flashing light before": "Frühere starke Reaktion auf Flackern oder Blitzlicht",
    "Neurological condition that may affect EEG or attention": "Neurologische Erkrankung, die EEG oder Aufmerksamkeit beeinflussen kann",
    "Migraine or light-sensitive headache history": "Migräne oder lichtempfindliche Kopfschmerzen in der Vorgeschichte",
    "Medication that may affect attention or EEG": "Medikamente, die Aufmerksamkeit oder EEG beeinflussen können",
    "Sleep last night in hours": "Schlaf letzte Nacht in Stunden",
    "Sleep quality from 0 to 6": "Schlafqualität von 0 bis 6",
    "Fatigue from 0 to 6": "Müdigkeit von 0 bis 6",
    "Alertness from 0 to 6": "Wachheit von 0 bis 6",
    "Stress from 0 to 6": "Stress von 0 bis 6",
    "Headache right now": "Aktuell Kopfschmerzen",
    "Eye strain right now": "Aktuell Augenbelastung",
    "Caffeine in last 4 hours": "Koffein in den letzten 4 Stunden",
    "Alcohol in last 12 hours": "Alkohol in den letzten 12 Stunden",
    "Medication today that may affect attention": "Heute Medikamente, die Aufmerksamkeit beeinflussen können",
    "Long screen use in the last hour": "Lange Bildschirmnutzung in der letzten Stunde",
    "Room light level from 0 to 6": "Raumhelligkeit von 0 bis 6",
    "Room light stayed stable during setup": "Raumlicht blieb während des Setups stabil",
    "Screen distance in cm": "Bildschirmabstand in cm",
    "Screen brightness in percent": "Bildschirmhelligkeit in Prozent",
    "Screen contrast in percent": "Bildschirmkontrast in Prozent",
    "Display refresh rate in Hz": "Bildwiederholrate in Hz",
    "Noise level from 0 to 6": "Geräuschpegel von 0 bis 6",
    "Room temperature (degree C)": "Raumtemperatur (Grad C)",
    "Interruptions expected during the session": "Unterbrechungen während der Sitzung erwartet",
    "Room notes": "Notizen zum Raum",
    "Favorite stimulus": "Bevorzugter Stimulus",
}

CHOICE_DE = {
    "English": "Englisch",
    "Deutsch": "Deutsch",
    "record": "record",
    "stream": "stream",
    "No": "Nein",
    "Yes": "Ja",
    "prefer not to say": "keine Angabe",
    "female": "weiblich",
    "male": "männlich",
    "diverse": "divers",
    "right": "rechts",
    "left": "links",
    "both": "beides",
    "none": "keine",
    "glasses": "Brille",
    "contact lenses": "Kontaktlinsen",
    "other": "andere",
}


def normalize_interface_language(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"de", "deutsch", "german", "ger", "deu"}:
        return LANG_DE
    return LANG_EN


def _ui(label: Any, language: str) -> str:
    text = str(label)
    if normalize_interface_language(language) == LANG_DE:
        return TEXT_DE.get(text, text)
    return text


def _choices_for_display(choices: Sequence[Any], language: str) -> tuple[list[str], dict[str, str]]:
    display: list[str] = []
    reverse: dict[str, str] = {}
    for choice in choices:
        raw = str(choice)
        shown = CHOICE_DE.get(raw, raw) if normalize_interface_language(language) == LANG_DE else raw
        display.append(shown)
        reverse[shown] = raw
    return display, reverse


def _dialog_values(dlg: Any, shown_data: Any) -> list[Any]:
    """Return PsychoPy dialog values across versions.

    Some PsychoPy versions return the values from ``Dlg.show()``. Others keep
    them only on ``dlg.data`` and return ``None``. Using this helper prevents
    edited runtime values from silently falling back to the defaults.
    """

    data = shown_data
    if data is None and hasattr(dlg, "data"):
        data = getattr(dlg, "data")
    if isinstance(data, dict):
        return list(data.values())
    return list(data) if data is not None else []


# ---------------------------------------------------------------------------
# Runtime parameter dialog
# ---------------------------------------------------------------------------

RUNTIME_PARAM_SPEC: Tuple[Dict[str, Any], ...] = (
    {"key": "proband_id",                   "label": "Participant ID",                   "type": "str"},
    {"key": "interface_language",           "label": "Interface language / Sprache",      "type": "choice",  "choices": ["English", "Deutsch"]},
    {"key": "acquisition_mode",             "label": "Acquisition mode",                 "type": "choice",  "choices": ["record", "stream"]},
    {"key": "trials_per_target",            "label": "Trials per target/frequency",      "type": "int",     "min": 1, "max": 100},
    {"key": "trial_duration_s",             "label": "Trial duration (s)",               "type": "float",   "min": 1.0, "max": 30.0},
    {"key": "cue_duration_s",               "label": "Cue duration (s)",                 "type": "float",   "min": 0.2, "max": 5.0},
    {"key": "target_spacing_deg",           "label": "Stimulus distance from center (deg)", "type": "float", "min": 1.0, "max": 30.0},
    {"key": "target_size_deg",              "label": "Stimulus size (deg)",              "type": "float",   "min": 0.5, "max": 20.0},
    {"key": "baseline_open_s",              "label": "Baseline eyes-open (s)",           "type": "float",   "min": 0.0, "max": 600.0},
    {"key": "include_eyes_closed_baseline", "label": "Include eyes-closed baseline?",    "type": "bool"},
    {"key": "baseline_closed_s",            "label": "Baseline eyes-closed (s)",         "type": "float",   "min": 0.0, "max": 600.0},
    {"key": "block_break_s",                "label": "Standard block break (s)",         "type": "float",   "min": 0.0, "max": 600.0},
    {"key": "long_break_s",                 "label": "Long break (s) - middle of study", "type": "float",   "min": 0.0, "max": 1200.0},
    {"key": "impedance_stable_s_initial",   "label": "Impedance stable hold - initial (s)", "type": "float", "min": 0.0, "max": 60.0},
    {"key": "impedance_stable_s_between",   "label": "Impedance stable hold - between (s)", "type": "float", "min": 0.0, "max": 60.0},
    {"key": "impedance_quality_threshold",  "label": "Impedance quality threshold (0..4)",  "type": "int",   "min": 0,   "max": 4},
    {"key": "idle_trials_per_source_block", "label": "Idle trials per source block",     "type": "int",     "min": 1, "max": 50},
    {"key": "seed_base",                    "label": "Random seed base",                 "type": "int",     "min": 0, "max": 10_000_000},
)


def run_runtime_parameter_dialog(
    defaults: Mapping[str, Any],
    *,
    title: str = "Main study - runtime parameters",
    headless: bool = False,
) -> Dict[str, Any]:
    """Show a dialog pre-populated with `defaults`, return edited values.

    `defaults` is a mapping from key -> value. Any key missing from
    `RUNTIME_PARAM_SPEC` is ignored; any spec key missing from `defaults`
    falls back to a type-appropriate sentinel.

    Returns a new dict with the same keys as `defaults`, with edited /
    coerced values. Raises RuntimeError if the operator cancels.
    """
    if headless:
        result = {k: defaults.get(k) for k in defaults}
        result["interface_language"] = normalize_interface_language(result.get("interface_language"))
        return result

    from psychopy import gui  # type: ignore

    language = normalize_interface_language(defaults.get("interface_language", LANG_EN))
    dlg = gui.Dlg(title=title)
    dlg.addText(
        "Review and confirm / edit the session parameters."
        if language == LANG_EN else
        "Bitte Sitzungsparameter prüfen und bestätigen."
    )

    order: List[Tuple[str, Dict[str, Any]]] = []
    for spec in RUNTIME_PARAM_SPEC:
        key = spec["key"]
        if key not in defaults:
            continue
        label = _ui(spec["label"], language)
        stype = spec["type"]
        initial = defaults[key]
        if key == "interface_language":
            initial = "Deutsch" if normalize_interface_language(initial) == LANG_DE else "English"
        if stype == "choice":
            choices, _reverse = _choices_for_display(list(spec["choices"]), language)
            display_initial = CHOICE_DE.get(str(initial), str(initial)) if language == LANG_DE else str(initial)
            if display_initial not in choices:
                choices = [display_initial] + [c for c in choices if c != display_initial]
            dlg.addField(label, initial=display_initial, choices=choices)
        elif stype == "bool":
            bool_choices = ["No", "Yes"] if not bool(initial) else ["Yes", "No"]
            if language == LANG_DE:
                bool_choices = [CHOICE_DE[c] for c in bool_choices]
            dlg.addField(label, initial=bool_choices[0], choices=bool_choices)
        else:
            dlg.addField(label, initial=initial)
        order.append((key, spec))

    ok_data = dlg.show()
    if not dlg.OK:
        raise RuntimeError("Runtime parameter dialog cancelled by operator.")

    data_values = _dialog_values(dlg, ok_data)

    result: Dict[str, Any] = dict(defaults)
    for (key, spec), raw in zip(order, data_values):
        if spec["type"] == "choice":
            _choices, reverse = _choices_for_display(list(spec.get("choices") or []), language)
            raw = reverse.get(str(raw), raw)
        result[key] = _coerce_value(raw, spec)
    result["interface_language"] = normalize_interface_language(result.get("interface_language"))
    return result


def _coerce_value(raw: Any, spec: Mapping[str, Any]) -> Any:
    stype = spec["type"]
    if raw is None or raw == "":
        if spec.get("optional"):
            return None
    try:
        if stype == "int":
            v = int(float(raw))
            if "min" in spec:
                v = max(int(spec["min"]), v)
            if "max" in spec:
                v = min(int(spec["max"]), v)
            return v
        if stype == "float":
            v = float(raw)
            if "min" in spec:
                v = max(float(spec["min"]), v)
            if "max" in spec:
                v = min(float(spec["max"]), v)
            return v
        if stype == "bool":
            return _to_bool(raw)
        if stype == "choice":
            if isinstance(raw, int):
                choices = spec.get("choices", [])
                if 0 <= raw < len(choices):
                    return str(choices[raw])
            return str(raw)
        # str / text
        return str(raw).strip()
    except (TypeError, ValueError):
        logger.warning("coerce failed | key=%s | raw=%r | type=%s", spec.get("key"), raw, stype)
        return raw


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


# ---------------------------------------------------------------------------
# Participant schema loader
# ---------------------------------------------------------------------------

def load_proband_schema(schema_path: Path | str) -> Dict[str, Any]:
    path = Path(schema_path)
    with open(path, "r", encoding="utf-8-sig") as fh:
        schema = json.load(fh)
    if not isinstance(schema, dict):
        raise ValueError(f"Participant schema must be a JSON object: {path}")
    return schema


# ---------------------------------------------------------------------------
# Pre-flight compliance checklist
# ---------------------------------------------------------------------------

def run_compliance_checklist(
    schema: Mapping[str, Any],
    *,
    title: str = "Pre-flight compliance",
    headless: bool = False,
    language: str = LANG_EN,
) -> Dict[str, Any]:
    items: Sequence[Mapping[str, Any]] = schema.get("pre_flight_compliance") or ()
    if not items:
        return {"items": {}, "all_required_ok": True, "skipped": True}

    if headless:
        return {
            "items": {str(it["key"]): True for it in items},
            "all_required_ok": True,
            "skipped": True,
        }

    from psychopy import gui  # type: ignore

    lang = normalize_interface_language(language)
    dlg = gui.Dlg(title=title if lang == LANG_EN else "Vorbereitungscheck")
    dlg.addText(
        "Tick every item that applies. Required items must all be checked to continue."
        if lang == LANG_EN else
        "Bitte alle zutreffenden Punkte bestätigen. Erforderliche Punkte müssen mit Ja beantwortet werden."
    )
    order: List[Mapping[str, Any]] = []
    for item in items:
        label = _ui(item.get("label", item.get("key", "?")), lang)
        required = bool(item.get("required"))
        suffix = "  (required)" if required and lang == LANG_EN else ("  (erforderlich)" if required else "")
        choices = ["No", "Yes"] if lang == LANG_EN else ["Nein", "Ja"]
        dlg.addField(label + suffix, initial=choices[0], choices=choices)
        order.append(item)
    data = dlg.show()
    if not dlg.OK:
        raise RuntimeError("Compliance checklist cancelled by operator.")

    data_values = _dialog_values(dlg, data)

    if len(data_values) != len(order):
        raise RuntimeError(
            f"Compliance dialog returned {len(data_values)} values, expected {len(order)}. "
            "PsychoPy version mismatch - please report."
        )

    answers: Dict[str, bool] = {}
    missing: List[str] = []
    for item, raw in zip(order, data_values):
        ok = _to_bool_from_yes_no(raw)
        answers[str(item["key"])] = ok
        if item.get("required") and not ok:
            missing.append(str(item.get("label", item["key"])))

    return {
        "items": answers,
        "missing_required": missing,
        "all_required_ok": not missing,
    }


def _to_bool_from_yes_no(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"yes", "y", "1", "true", "on", "ja", "j"}


# ---------------------------------------------------------------------------
# Schema-driven questionnaire sections
# ---------------------------------------------------------------------------

def run_questionnaire_section(
    schema: Mapping[str, Any],
    section_key: str,
    *,
    title_override: Optional[str] = None,
    headless: bool = False,
    language: str = LANG_EN,
) -> Dict[str, Any]:
    section = schema.get(section_key)
    if not section:
        return {}
    fields: Sequence[Mapping[str, Any]] = section.get("fields") or ()
    if not fields:
        return {}

    if headless:
        return {str(f["key"]): _fallback_headless_value(f) for f in fields}

    from psychopy import gui  # type: ignore

    lang = normalize_interface_language(language)
    section_title = str(section.get("title") or section_key)
    dlg = gui.Dlg(title=title_override or _ui(section_title, lang))
    dlg.addText(_ui(section_title, lang))
    order: List[Mapping[str, Any]] = []
    choice_reverse_by_key: Dict[str, Dict[str, str]] = {}
    for field in fields:
        label = _ui(field.get("label", field.get("key", "?")), lang)
        ftype = field.get("type", "text")
        optional = bool(field.get("optional"))
        suffix = "" if not optional else "  (optional)"
        if ftype == "choice":
            choices, reverse = _choices_for_display(list(field.get("choices") or []), lang)
            choice_reverse_by_key[str(field["key"])] = reverse
            initial_choice = choices[0] if choices else ""
            dlg.addField(label + suffix, initial=initial_choice, choices=choices)
        elif ftype == "bool":
            choices = ["No", "Yes"] if lang == LANG_EN else ["Nein", "Ja"]
            dlg.addField(label + suffix, initial=choices[0], choices=choices)
        else:
            dlg.addField(label + suffix, initial="")
        order.append(field)

    data = dlg.show()
    if not dlg.OK:
        raise RuntimeError(f"Questionnaire section '{section_key}' cancelled by operator.")

    data_values = _dialog_values(dlg, data)

    answers: Dict[str, Any] = {}
    for field, raw in zip(order, data_values):
        key = str(field["key"])
        if field.get("type") == "choice":
            raw = choice_reverse_by_key.get(key, {}).get(str(raw), raw)
        answers[key] = _coerce_field_value(raw, field)
    return answers


def _fallback_headless_value(field: Mapping[str, Any]) -> Any:
    ftype = field.get("type", "text")
    if ftype == "bool":
        return False
    if ftype == "choice":
        choices = field.get("choices") or [""]
        return choices[0]
    if ftype == "int":
        return field.get("min", 0)
    if ftype == "float":
        return float(field.get("min", 0.0))
    return ""


def _coerce_field_value(raw: Any, field: Mapping[str, Any]) -> Any:
    ftype = field.get("type", "text")
    optional = bool(field.get("optional"))
    if (raw is None or (isinstance(raw, str) and not raw.strip())) and optional:
        return None
    try:
        if ftype == "int":
            if raw == "" or raw is None:
                return None if optional else int(field.get("min", 0))
            v = int(float(raw))
            if "min" in field:
                v = max(int(field["min"]), v)
            if "max" in field:
                v = min(int(field["max"]), v)
            return v
        if ftype == "float":
            if raw == "" or raw is None:
                return None if optional else float(field.get("min", 0.0))
            v = float(raw)
            if "min" in field:
                v = max(float(field["min"]), v)
            if "max" in field:
                v = min(float(field["max"]), v)
            return v
        if ftype == "bool":
            return _to_bool_from_yes_no(raw)
        if ftype == "choice":
            return str(raw)
        return str(raw).strip()
    except (TypeError, ValueError):
        logger.warning("field coerce failed | key=%s | raw=%r | type=%s",
                       field.get("key"), raw, ftype)
        return raw


# ---------------------------------------------------------------------------
# Exclusion logic
# ---------------------------------------------------------------------------

def check_exclusion_criteria(
    schema: Mapping[str, Any],
    q1_answers: Mapping[str, Any],
) -> List[Tuple[str, str]]:
    """Return a list of (field_key, reason) pairs for hard exclusion hits."""
    excl = (schema.get("exclusion_criteria") or {}).get("hard") or []
    hits: List[Tuple[str, str]] = []
    for key in excl:
        val = q1_answers.get(key)
        if bool(val):
            hits.append((str(key), f"{key} = True (hard exclusion)"))
    return hits


__all__ = [
    "RUNTIME_PARAM_SPEC",
    "normalize_interface_language",
    "run_runtime_parameter_dialog",
    "load_proband_schema",
    "run_compliance_checklist",
    "run_questionnaire_section",
    "check_exclusion_criteria",
]


if __name__ == "__main__":
    schema_path = Path("configs") / "main_study_proband_schema.json"
    schema = load_proband_schema(schema_path)
    q1 = run_questionnaire_section(schema, "q1_trait", headless=True)
    compliance = run_compliance_checklist(schema, headless=True)

    print("Main study questionnaires module demo")
    print("This demo does not open PsychoPy dialogs.")
    print(f"Schema version: {schema.get('schema_version')}")
    print(f"Headless q1 fields: {len(q1)}")
    print(f"Compliance ok: {compliance['all_required_ok']}")

