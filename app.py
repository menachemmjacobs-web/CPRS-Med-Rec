import json
import os
import re
from dataclasses import dataclass
from html import escape
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import streamlit as st


HIGH_RISK_TERMS = {
    "anticoagulant": {
        "warfarin",
        "apixaban",
        "eliquis",
        "rivaroxaban",
        "xarelto",
        "dabigatran",
        "pradaxa",
        "edoxaban",
        "savaysa",
        "enoxaparin",
        "lovenox",
        "heparin",
    },
    "insulin or hypoglycemic": {
        "insulin",
        "glargine",
        "lantus",
        "basaglar",
        "detemir",
        "levemir",
        "degludec",
        "tresiba",
        "lispro",
        "humalog",
        "aspart",
        "novolog",
        "glipizide",
        "glyburide",
        "glimepiride",
    },
    "opioid": {
        "morphine",
        "oxycodone",
        "hydrocodone",
        "hydromorphone",
        "fentanyl",
        "methadone",
        "tramadol",
        "buprenorphine",
    },
    "antiplatelet": {"clopidogrel", "plavix", "ticagrelor", "brilinta", "prasugrel", "effient"},
    "digoxin": {"digoxin"},
    "antiarrhythmic": {"amiodarone", "sotalol", "dofetilide", "flecainide", "propafenone"},
    "antiepileptic": {"phenytoin", "carbamazepine", "valproate", "divalproex", "levetiracetam", "lamotrigine"},
    "immunosuppressant": {"tacrolimus", "cyclosporine", "mycophenolate", "azathioprine", "methotrexate"},
    "sedative": {"alprazolam", "lorazepam", "clonazepam", "diazepam", "zolpidem", "temazepam"},
}

DOSE_PATTERN = re.compile(
    r"(?P<dose>\d+(?:\.\d+)?)\s*(?P<unit>mg|mcg|g|gm|units?|iu|meq|ml|%)\s*(?=\W|$)",
    re.IGNORECASE,
)
FREQUENCY_PATTERN = re.compile(
    r"\b("
    r"daily|once daily|twice daily|three times daily|four times daily|"
    r"once a day|twice a day|three times a day|four times a day|every morning|every evening|"
    r"qday|qd|bid|tid|qid|qhs|qam|qpm|qod|weekly|monthly|"
    r"q\d+\s*(?:h|hours?|days?|weeks?|months?)?|every\s+\d+\s+(?:hours?|days?|weeks?|months?)|as needed|prn"
    r")\b",
    re.IGNORECASE,
)
SECTION_SPLIT_PATTERN = re.compile(r"[\n;]+")
TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9-]*")
NUMBERED_MED_PATTERN = re.compile(r"^\s*\d+\)\s+(?P<med>.+)$")
ROUTE_PATTERN = re.compile(
    r"\b(po|iv|ivpb|sq|subq|sc|im|top|ext|inh|sl|oph|otic|patch|tablet|tab|capsule|cap|injection|inj)\b",
    re.IGNORECASE,
)
MED_CONTINUATION_STOP_PATTERN = re.compile(
    r"^\s*(indication:|status$|=+|active .*medications|current active|discharged to:|"
    r"nursing note|medication management after discharge:|\d+\.\s)",
    re.IGNORECASE,
)
IGNORE_LINE_PATTERN = re.compile(
    r"\b("
    r"active inpatient medications|active outpatient|clinic medications|current active|"
    r"discharged to|counseling|appointment|supplies|wound care|catheters|"
    r"question|tobacco cessation|bmi|crcl|protocol|obtain wipes|indication:"
    r")\b",
    re.IGNORECASE,
)
NAME_STOP_PATTERN = re.compile(
    r"\b("
    r"tab|tablet|cap|capsule|soln|solution|inj|injection|gel|patch|drop|drops|"
    r"take|apply|instill|administer|infuse|give|use"
    r")\b",
    re.IGNORECASE,
)
NAME_MODIFIER_TOKENS = {
    "besylate",
    "sodium",
    "na",
    "maleate",
    "hcl",
    "hydrochloride",
    "oral",
    "top",
    "oph",
    "soln",
    "solution",
    "inj",
    "injection",
    "gel",
    "patch",
    "tab",
    "tablet",
    "cap",
    "capsule",
}


@dataclass(frozen=True)
class Medication:
    name: str
    display_name: str
    dose: str
    frequency: str
    source_line: str


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def clean_line(line: str) -> str:
    line = line.replace("\xa0", " ")
    line = re.sub(r"^\s*(?:[-*]|\d+\))\s*", "", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def medication_display_name_from_line(line: str) -> str:
    dose_match = DOSE_PATTERN.search(line)
    stop = dose_match.start() if dose_match else len(line)
    prefix = line[:stop]
    prefix = NAME_STOP_PATTERN.split(prefix)[0]
    prefix = prefix.replace("_", " ")
    tokens = TOKEN_PATTERN.findall(prefix.lower())

    skipped = {"continue", "start", "stop", "hold", "home", "med", "medication", "take", "use"}
    tokens = [token for token in tokens if token not in skipped]
    if not tokens:
        tokens = TOKEN_PATTERN.findall(line.lower())[:2]

    return normalize_name(" ".join(tokens[:4]))


def canonical_medication_name(display_name: str) -> str:
    tokens = [token for token in TOKEN_PATTERN.findall(display_name.lower()) if token not in NAME_MODIFIER_TOKENS]
    if not tokens:
        return normalize_name(display_name)
    if tokens[0] in {"piperacillin", "sulfamethoxazole", "trimethoprim"} and len(tokens) > 1:
        return normalize_name(" ".join(tokens[:2]))
    return normalize_name(tokens[0])


def extract_dose(line: str) -> str:
    match = DOSE_PATTERN.search(line)
    if not match:
        return ""
    return f"{match.group('dose')} {match.group('unit').lower()}"


def normalize_frequency(freq: str) -> str:
    normalized = re.sub(r"\s+", " ", freq.lower()).strip()
    aliases = {
        "qd": "daily",
        "qday": "daily",
        "once daily": "daily",
        "once a day": "daily",
        "bid": "twice daily",
        "twice a day": "twice daily",
        "tid": "three times daily",
        "three times a day": "three times daily",
        "qid": "four times daily",
        "four times a day": "four times daily",
        "every morning": "qam",
        "every evening": "qpm",
        "prn": "as needed",
    }
    return aliases.get(normalized, normalized)


def extract_frequency(line: str) -> str:
    match = FREQUENCY_PATTERN.search(line)
    return normalize_frequency(match.group(1)) if match else ""


def high_risk_categories_for_text(text: str) -> Set[str]:
    text_lower = text.lower()
    return {
        category
        for category, terms in HIGH_RISK_TERMS.items()
        if any(term in text_lower for term in terms)
    }


def has_medication_signal(line: str) -> bool:
    if "?" in line:
        return False
    if IGNORE_LINE_PATTERN.search(line) and not DOSE_PATTERN.search(line):
        return False
    return bool(
        DOSE_PATTERN.search(line)
        or FREQUENCY_PATTERN.search(line)
        or ROUTE_PATTERN.search(line)
        or high_risk_categories_for_text(line)
    )


def medication_blocks(text: str) -> List[str]:
    blocks: List[str] = []
    current: List[str] = []
    saw_numbered_entry = False

    for raw_line in text.splitlines():
        line = clean_line(raw_line)
        if not line:
            continue

        numbered_match = NUMBERED_MED_PATTERN.match(raw_line.replace("\xa0", " "))
        if numbered_match:
            saw_numbered_entry = True
            if current:
                blocks.append(" ".join(current))
            current = [clean_line(numbered_match.group("med"))]
            continue

        if current:
            if MED_CONTINUATION_STOP_PATTERN.search(line):
                continue
            current.append(line)
            continue

    if current:
        blocks.append(" ".join(current))

    if saw_numbered_entry:
        return blocks

    return [clean_line(raw_line) for raw_line in SECTION_SPLIT_PATTERN.split(text) if clean_line(raw_line)]


def parse_medications(text: str) -> Dict[str, Medication]:
    meds: Dict[str, Medication] = {}
    for line in medication_blocks(text):
        if not line or not has_medication_signal(line):
            continue

        display_name = medication_display_name_from_line(line)
        name = canonical_medication_name(display_name)
        if not name:
            continue

        meds[name] = Medication(
            name=name,
            display_name=display_name.title(),
            dose=extract_dose(line),
            frequency=extract_frequency(line),
            source_line=line,
        )
    return meds


def detect_high_risk(meds: Iterable[Medication], notes: str) -> Dict[str, Set[str]]:
    notes_lower = notes.lower()
    findings: Dict[str, Set[str]] = {}

    for med in meds:
        med_text = f"{med.name} {med.source_line}".lower()
        categories = high_risk_categories_for_text(med_text)

        note_flags = []
        if med.name and med.name in notes_lower:
            for flag in ("renal", "kidney", "bleeding", "fall", "delirium", "hypoglycemia", "supratherapeutic"):
                if flag in notes_lower:
                    note_flags.append(f"note mentions {flag}")
        categories.update(note_flags)

        if categories:
            findings[med.name] = categories

    return findings


def describe_change(old: Medication, new: Medication) -> Tuple[Optional[str], Optional[str]]:
    dose_change = None
    frequency_change = None

    if old.dose and new.dose and old.dose != new.dose:
        dose_change = f"{old.dose} -> {new.dose}"
    if old.frequency and new.frequency and old.frequency != new.frequency:
        frequency_change = f"{old.frequency} -> {new.frequency}"

    return dose_change, frequency_change


def has_discharge_list(discharge_text: str) -> bool:
    return bool(parse_medications(discharge_text))


def build_analysis(
    admission_text: str,
    inpatient_text: str,
    discharge_text: str,
    notes_text: str,
) -> pd.DataFrame:
    admission = parse_medications(admission_text)
    inpatient = parse_medications(inpatient_text)
    discharge = parse_medications(discharge_text)
    discharge_available = bool(discharge)
    active_names = set(admission) | set(inpatient) | set(discharge)
    high_risk = detect_high_risk(
        [*admission.values(), *inpatient.values(), *discharge.values()],
        notes_text,
    )

    rows: List[dict] = []

    for name in sorted(active_names):
        admission_med = admission.get(name)
        inpatient_med = inpatient.get(name)
        discharge_med = discharge.get(name)
        reference_med = admission_med or inpatient_med or discharge_med
        statuses = []
        details = []

        if discharge_available and name not in admission and name in discharge:
            statuses.append("New at discharge")
            details.append("Not on the admission list.")

        if discharge_available and name not in admission and name in inpatient and name not in discharge:
            statuses.append("Inpatient med not continued")
            details.append("Used inpatient only. Confirm it should stop.")

        if discharge_available and name in admission and name not in discharge:
            statuses.append("Home med stopped")
            details.append("Home medication is not on the discharge list.")

        if not discharge_available and name in admission and name not in inpatient:
            statuses.append("Held on admission")
            details.append("Home medication is not active on the inpatient list.")

        if not discharge_available and name not in admission and name in inpatient:
            statuses.append("New inpatient med")
            details.append("Medication appears inpatient but not on admission med rec.")

        old_for_change = admission_med or inpatient_med
        new_for_change = discharge_med if discharge_available else inpatient_med
        if old_for_change and new_for_change:
            dose_change, frequency_change = describe_change(old_for_change, new_for_change)
            if dose_change:
                statuses.append("Dose change")
                details.append(dose_change)
            if frequency_change:
                statuses.append("Frequency change")
                details.append(frequency_change)

        if name in high_risk:
            statuses.append("High-risk med")
            details.append(", ".join(sorted(high_risk[name])))

        if statuses:
            rows.append(
                {
                    "Medication": reference_med.display_name if reference_med else name.title(),
                    "Category": "; ".join(dict.fromkeys(statuses)),
                    "Admission": admission_med.source_line if admission_med else "",
                    "Inpatient": inpatient_med.source_line if inpatient_med else "",
                    "Discharge": discharge_med.source_line if discharge_med else "",
                    "Details": " | ".join(dict.fromkeys(details)),
                    "Review Priority": "High" if name in high_risk else "Standard",
                }
            )

    return pd.DataFrame(
        rows,
        columns=["Medication", "Category", "Admission", "Inpatient", "Discharge", "Details", "Review Priority"],
    )


def load_example() -> Tuple[str, str, str, str]:
    return (
        "Lisinopril 10 mg daily\nMetformin 500 mg twice daily\nWarfarin 5 mg daily\nAtorvastatin 20 mg qhs",
        "Lisinopril 20 mg daily\nInsulin glargine 10 units qhs\nAtorvastatin 20 mg qhs\nHeparin 5000 units q8h",
        "Lisinopril 20 mg daily\nInsulin glargine 12 units qhs\nAtorvastatin 40 mg qhs\nApixaban 5 mg twice daily",
        "Renal function fluctuated during admission. History of fall risk. Anticoagulation changed after INR review.",
    )


def split_categories(value: str) -> List[str]:
    return [category.strip() for category in value.split(";") if category.strip()]


def action_label(row: pd.Series) -> str:
    categories = split_categories(str(row["Category"]))
    if "Held on admission" in categories:
        return "Held on admission"
    if "New inpatient med" in categories:
        return "New inpatient med"
    if "Home med stopped" in categories:
        return "Confirm intentional stop"
    if "Inpatient med not continued" in categories:
        return "Confirm no discharge prescription needed"
    if "New at discharge" in categories:
        return "Confirm indication, dose, and duration"
    if "Dose change" in categories or "Frequency change" in categories:
        return "Confirm changed regimen"
    if "High-risk med" in categories:
        return "Check before discharge"
    return "Review"


def medication_state(value: str) -> str:
    return "Present" if str(value).strip() else "Absent"


def medication_regimen(med: Optional[Medication]) -> str:
    if not med:
        return ""
    parts = [part for part in (med.dose, med.frequency) if part]
    return " ".join(parts)


def build_timeline(admission_text: str, inpatient_text: str, discharge_text: str, analysis: pd.DataFrame) -> pd.DataFrame:
    admission = parse_medications(admission_text)
    inpatient = parse_medications(inpatient_text)
    discharge = parse_medications(discharge_text)
    issue_lookup = {row["Medication"]: row for _, row in analysis.iterrows()}
    rows = []

    for name in sorted(set(admission) | set(inpatient) | set(discharge)):
        med = admission.get(name) or inpatient.get(name) or discharge.get(name)
        issue_row = issue_lookup.get(med.display_name)
        rows.append(
            {
                "Medication": med.display_name,
                "Admission": medication_state(admission.get(name).source_line if name in admission else ""),
                "Admission Regimen": medication_regimen(admission.get(name)),
                "Inpatient": medication_state(inpatient.get(name).source_line if name in inpatient else ""),
                "Inpatient Regimen": medication_regimen(inpatient.get(name)),
                "Discharge": medication_state(discharge.get(name).source_line if name in discharge else ""),
                "Discharge Regimen": medication_regimen(discharge.get(name)),
                "Transition": action_label(issue_row) if issue_row is not None else "No flagged transition",
                "Priority": issue_row["Review Priority"] if issue_row is not None else "Standard",
            }
        )

    return pd.DataFrame(rows)


def summarize_categories(analysis: pd.DataFrame) -> Dict[str, int]:
    summary = {
        "High Risk": 0,
        "New": 0,
        "Inpatient Only": 0,
        "Discontinued": 0,
        "Held": 0,
        "New Inpatient": 0,
        "Dose": 0,
        "Frequency": 0,
    }
    if analysis.empty:
        return summary

    category_text = analysis["Category"].str.lower()
    summary["High Risk"] = int(category_text.str.contains("high-risk").sum())
    summary["New"] = int(category_text.str.contains("new at discharge").sum())
    summary["Inpatient Only"] = int(category_text.str.contains("inpatient med not continued").sum())
    summary["Discontinued"] = int(category_text.str.contains("home med stopped").sum())
    summary["Held"] = int(category_text.str.contains("held on admission").sum())
    summary["New Inpatient"] = int(category_text.str.contains("new inpatient med").sum())
    summary["Dose"] = int(category_text.str.contains("dose change").sum())
    summary["Frequency"] = int(category_text.str.contains("frequency change").sum())
    return summary


def rows_with_category(analysis: pd.DataFrame, category: str, include_high_risk: bool = True) -> pd.DataFrame:
    if analysis.empty:
        return analysis
    mask = analysis["Category"].str.contains(category, case=False, regex=False)
    if not include_high_risk:
        mask = mask & (analysis["Review Priority"] != "High")
    return analysis[mask]


def parser_diagnostics(admission_text: str, inpatient_text: str, discharge_text: str, timeline: pd.DataFrame) -> Dict[str, Any]:
    parsed_counts = {
        "Admission": len(parse_medications(admission_text)),
        "Inpatient": len(parse_medications(inpatient_text)),
        "Discharge": len(parse_medications(discharge_text)),
    }
    suspect_terms = {
        "dose",
        "per",
        "pharmacy",
        "omni",
        "active",
        "current",
        "status",
        "normal",
        "saline",
    }
    suspect_names = []
    if not timeline.empty:
        for medication in timeline["Medication"].tolist():
            tokens = set(TOKEN_PATTERN.findall(str(medication).lower()))
            if tokens & suspect_terms or len(tokens) >= 5:
                suspect_names.append(medication)

    return {
        "parsed_counts": parsed_counts,
        "suspect_names": sorted(set(suspect_names)),
        "confidence": "Review parser output" if suspect_names else "Looks clean",
    }


def resident_signout(summary: Dict[str, int], analysis: pd.DataFrame) -> str:
    has_discharge_review = bool(summary["New"] or summary["Discontinued"] or summary["Inpatient Only"])
    title = "Discharge med rec signout:" if has_discharge_review else "Admission hold review signout:"
    lines = [
        title,
        f"- High-risk: {summary['High Risk']}; new at discharge: {summary['New']}; "
        f"home meds stopped: {summary['Discontinued']}; inpatient-only: {summary['Inpatient Only']}; "
        f"held on admission: {summary['Held']}; new inpatient: {summary['New Inpatient']}.",
    ]

    if analysis.empty:
        lines.append("- No transition issues were flagged by the local parser.")
        return "\n".join(lines)

    high_risk = analysis[analysis["Review Priority"] == "High"]
    if not high_risk.empty:
        meds = ", ".join(high_risk["Medication"].tolist())
        lines.append(f"- Check before discharge: {meds}.")

    stopped = rows_with_category(analysis, "Home med stopped")
    if not stopped.empty:
        meds = ", ".join(stopped["Medication"].tolist())
        lines.append(f"- Confirm intentional stop: {meds}.")

    held = rows_with_category(analysis, "Held on admission")
    if not held.empty:
        meds = ", ".join(held["Medication"].tolist())
        lines.append(f"- Home meds held on admission: {meds}.")

    new_inpatient = rows_with_category(analysis, "New inpatient med", include_high_risk=False)
    if not new_inpatient.empty:
        meds = ", ".join(new_inpatient["Medication"].tolist())
        lines.append(f"- New inpatient meds not on admission med rec: {meds}.")

    new_discharge = rows_with_category(analysis, "New at discharge")
    if not new_discharge.empty:
        meds = ", ".join(new_discharge["Medication"].tolist())
        lines.append(f"- Confirm indication, dose, and duration for new discharge meds: {meds}.")

    inpatient_only = rows_with_category(analysis, "Inpatient med not continued", include_high_risk=False)
    if not inpatient_only.empty:
        meds = ", ".join(inpatient_only["Medication"].tolist())
        lines.append(f"- Inpatient meds not continued: {meds}.")

    changed = analysis[
        analysis["Category"].str.contains("Dose change|Frequency change", case=False, regex=True)
    ]
    if not changed.empty:
        meds = ", ".join(changed["Medication"].tolist())
        lines.append(f"- Confirm changed regimen: {meds}.")

    lines.append("- Verify against CPRS orders, indications, renal function, allergies, and active medication orders.")
    return "\n".join(lines)


def status_class(value: str) -> str:
    if value == "Present":
        return "status-present"
    return "status-absent"


def transition_class(value: str) -> str:
    value_lower = value.lower()
    if "check" in value_lower:
        return "transition-high"
    if "stop" in value_lower or "prescription" in value_lower:
        return "transition-review"
    if "changed" in value_lower or "duration" in value_lower:
        return "transition-change"
    return "transition-ok"


def render_timeline(timeline: pd.DataFrame) -> None:
    if timeline.empty:
        st.info("No medications were parsed for the timeline.")
        return

    rows_html = []
    for _, row in timeline.iterrows():
        rows_html.append(
            "<tr>"
            f"<td class='timeline-med'>{escape(str(row['Medication']))}</td>"
            f"<td><span class='status-pill {status_class(str(row['Admission']))}'>{escape(str(row['Admission']))}</span></td>"
            f"<td><span class='status-pill {status_class(str(row['Inpatient']))}'>{escape(str(row['Inpatient']))}</span></td>"
            f"<td><span class='status-pill {status_class(str(row['Discharge']))}'>{escape(str(row['Discharge']))}</span></td>"
            f"<td><span class='transition-pill {transition_class(str(row['Transition']))}'>{escape(str(row['Transition']))}</span></td>"
            f"<td>{escape(str(row['Priority']))}</td>"
            "</tr>"
        )

    st.markdown(
        f"""
        <div class="timeline-wrap">
            <table class="timeline-table">
                <thead>
                    <tr>
                        <th>Medication</th>
                        <th>Admission</th>
                        <th>Inpatient</th>
                        <th>Discharge</th>
                        <th>Review</th>
                        <th>Priority</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows_html)}
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def board_action_label(row: Dict[str, Any], discharge_available: bool) -> str:
    admission = row["Admission"] == "Present"
    inpatient = row["Inpatient"] == "Present"
    discharge = row["Discharge"] == "Present"
    transition = str(row["Transition"])

    if not discharge_available:
        if transition == "Confirm changed regimen":
            return "Dose/frequency changed"
        if admission and inpatient:
            return "Continued inpatient"
        if admission and not inpatient:
            return "Held on admission"
        if not admission and inpatient:
            return "New inpatient"
        return "Review"

    if transition == "Held on admission":
        return "Held on admission"
    if transition == "New inpatient med":
        return "New inpatient"
    if admission and not discharge:
        return "Stopped at discharge"
    if not admission and inpatient and not discharge:
        return "Hospital only"
    if not admission and discharge:
        return "New at discharge"
    if transition == "Confirm changed regimen":
        return "Dose/frequency changed"
    if discharge:
        return "Continued"
    return "Review"


def board_subtext(row: Dict[str, Any], discharge_available: bool) -> str:
    discharge = row["Discharge"] == "Present"
    transition = str(row["Transition"])
    action = board_action_label(row, discharge_available)
    high_risk_prefix = "High-risk. " if row["Priority"] == "High" else ""

    messages = {
        "Held on admission": "Home medication is not active on inpatient orders.",
        "New inpatient": "Active inpatient but not listed on admission med rec.",
        "Continued inpatient": "Home medication is active on inpatient orders.",
        "Stopped at discharge": "Home medication is not continued on discharge.",
        "Hospital only": "Used in the hospital and absent from discharge meds.",
        "New at discharge": "New discharge medication. Verify indication, dose, and duration.",
        "Dose/frequency changed": "Continues with a dose or frequency change.",
        "Continued": "Medication continues at discharge.",
    }
    message = messages.get(action, "Verify plan before signing." if not discharge else "Review discharge plan.")
    return high_risk_prefix + message


def render_medication_board(timeline: pd.DataFrame, discharge_available: bool) -> None:
    if timeline.empty:
        st.info("No medications were parsed for the medication board.")
        return

    rows = timeline.sort_values("Medication").to_dict(orient="records")
    row_html = []
    for row in rows:
        transition = str(row["Transition"])
        action_label = board_action_label(row, discharge_available)
        priority_class = "board-action-high" if row["Priority"] == "High" else transition_class(transition)
        row_class = "recon-row recon-row-high" if row["Priority"] == "High" else "recon-row"

        cells = []
        labels = {
            "Admission": "Admission Med Rec",
            "Inpatient": "Inpatient Meds",
            "Discharge": "Proposed DC Med Rec",
        }
        for column in ("Admission", "Inpatient", "Discharge"):
            present = row[column] == "Present"
            cell_class = "recon-cell recon-present" if present else "recon-cell recon-absent"
            regimen = str(row.get(f"{column} Regimen", "")).strip()
            if present:
                regimen_html = f"<span class='recon-dose'>{escape(regimen)}</span>" if regimen else ""
                cell_html = f"<span class='recon-med'>{escape(str(row['Medication']))}</span>{regimen_html}"
            else:
                cell_html = "<span class='recon-empty'>-</span>"
            cells.append(f"<div class='{cell_class}' data-label='{labels[column]}'>{cell_html}</div>")

        row_html.append(
            f"<div class='{row_class}'>"
            f"{''.join(cells)}"
            "<div class='recon-review'>"
            f"<span class='board-action {priority_class}'>{escape(action_label)}</span>"
            f"<span class='recon-reason'>{escape(board_subtext(row, discharge_available))}</span>"
            "</div>"
            "</div>"
        )

    discharge_header = "Proposed DC Med Rec" if discharge_available else "Proposed DC Med Rec (not provided)"
    board_html = (
        "<div class='board-legend'>"
        "<span><b>Goal:</b> each row is one medication across the three med lists.</span>"
        "<span><span class='legend-dot legend-dot-on'></span> Present</span>"
        "<span><span class='legend-dot legend-dot-off'></span> Absent</span>"
        "</div>"
        "<div class='recon-board'>"
        "<div class='recon-header'>Admission Med Rec</div>"
        "<div class='recon-header'>Inpatient Meds</div>"
        f"<div class='recon-header'>{discharge_header}</div>"
        "<div class='recon-header recon-review-header'>Review</div>"
        f"{''.join(row_html)}"
        "</div>"
    )
    st.markdown(board_html, unsafe_allow_html=True)


def board_cell_text(row: Dict[str, Any], column: str) -> str:
    if row[column] != "Present":
        return "-"
    regimen = str(row.get(f"{column} Regimen", "")).strip()
    medication = str(row["Medication"])
    return f"{medication} ({regimen})" if regimen else medication


def board_export_dataframe(timeline: pd.DataFrame, discharge_available: bool) -> pd.DataFrame:
    discharge_header = "Proposed DC Med Rec" if discharge_available else "Proposed DC Med Rec (not provided)"
    rows = []

    for row in timeline.sort_values("Medication").to_dict(orient="records"):
        rows.append(
            {
                "Admission Med Rec": board_cell_text(row, "Admission"),
                "Inpatient Meds": board_cell_text(row, "Inpatient"),
                discharge_header: board_cell_text(row, "Discharge"),
                "Review": board_action_label(row, discharge_available),
                "Review Details": board_subtext(row, discharge_available),
                "Priority": row["Priority"],
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "Admission Med Rec",
            "Inpatient Meds",
            discharge_header,
            "Review",
            "Review Details",
            "Priority",
        ],
    )


def get_openai_api_key() -> str:
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        secret_key = ""
    return str(secret_key or os.getenv("OPENAI_API_KEY", "")).strip()


def llm_model_name() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-5.2")


def analysis_payload(analysis: pd.DataFrame, timeline: pd.DataFrame, summary: Dict[str, int]) -> Dict[str, Any]:
    action_rows = []
    for _, row in analysis.iterrows():
        action_rows.append(
            {
                "medication": row["Medication"],
                "issues": split_categories(str(row["Category"])),
                "action": action_label(row),
                "priority": row["Review Priority"],
                "details": row["Details"],
                "admission": row["Admission"],
                "inpatient": row["Inpatient"],
                "discharge": row["Discharge"],
            }
        )

    return {
        "summary": summary,
        "action_list": action_rows,
        "timeline": timeline.to_dict(orient="records") if not timeline.empty else [],
    }


def call_llm(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=get_openai_api_key())
    response = client.responses.create(
        model=llm_model_name(),
        store=False,
        instructions=(
            "You are a careful internal medicine resident helping review a de-identified medication "
            "transition report. Do not diagnose, prescribe, or invent missing facts. Only comment on "
            "the structured medication transition data provided. Use concise, clinical language. "
            "Always remind the user to verify against CPRS/orders before acting."
        ),
        input=prompt,
    )
    return response.output_text.strip()


def llm_review_prompt(payload: Dict[str, Any]) -> str:
    return (
        "Review this de-identified medication transition report for sanity. Return:\n"
        "1. A 3-5 bullet resident signout of what needs attention.\n"
        "2. Any apparent parser weirdness or medication names that look suspect.\n"
        "3. The highest-risk item to verify before discharge.\n"
        "4. A short list of questions the clinician should answer in CPRS.\n\n"
        f"Structured report JSON:\n{json.dumps(payload, indent=2)}"
    )


def llm_question_prompt(payload: Dict[str, Any], review: str, question: str) -> str:
    return (
        "Answer the clinician's question using only this structured de-identified medication transition "
        "report and the prior LLM sanity check. If the question cannot be answered from the data, say what "
        "to verify in CPRS. Keep the answer brief and action-oriented.\n\n"
        f"Structured report JSON:\n{json.dumps(payload, indent=2)}\n\n"
        f"Prior sanity check:\n{review}\n\n"
        f"Clinician question:\n{question}"
    )


def render_llm_review(payload: Dict[str, Any]) -> None:
    st.markdown("#### LLM Sanity Check")
    st.caption("Optional. Sends the structured, de-identified med transition output to the OpenAI API.")

    if not get_openai_api_key():
        st.info(
            "Set `OPENAI_API_KEY` in your environment or Streamlit secrets to enable the sanity check "
            "and interactive questions."
        )
        return

    phi_confirmed = st.checkbox(
        "I confirm the text entered above is de-identified and contains no PHI.",
        key="phi_confirmed_for_llm",
    )
    if not phi_confirmed:
        st.warning("Confirm de-identification before sending anything to the LLM.")
        return

    if "llm_review" not in st.session_state:
        st.session_state.llm_review = ""
    if "llm_answers" not in st.session_state:
        st.session_state.llm_answers = []

    if st.button("Run LLM sanity check", type="secondary", width="stretch"):
        with st.spinner("Reviewing structured transition output..."):
            try:
                st.session_state.llm_review = call_llm(llm_review_prompt(payload))
                st.session_state.llm_answers = []
            except Exception as exc:
                st.error(f"LLM review failed: {exc}")

    if st.session_state.llm_review:
        st.markdown(st.session_state.llm_review)

        question = st.text_input(
            "Ask a follow-up about this med transition",
            placeholder="Example: What should I verify before signing discharge meds?",
        )
        if st.button("Ask LLM", width="stretch") and question.strip():
            with st.spinner("Checking the structured report..."):
                try:
                    answer = call_llm(llm_question_prompt(payload, st.session_state.llm_review, question.strip()))
                    st.session_state.llm_answers.append((question.strip(), answer))
                except Exception as exc:
                    st.error(f"LLM question failed: {exc}")

        for asked, answer in st.session_state.llm_answers:
            st.markdown(f"**Q:** {escape(asked)}")
            st.markdown(answer)


def render_issue_card(row: pd.Series) -> None:
    priority_class = "priority-high" if row["Review Priority"] == "High" else "priority-standard"
    category_html = "".join(
        f"<span class='issue-pill'>{escape(category)}</span>"
        for category in split_categories(row["Category"])
    )
    source_rows = []
    for label in ("Admission", "Inpatient", "Discharge"):
        value = str(row[label]).strip()
        if value:
            source_rows.append(
                f"<div class='source-row'><span>{label}</span><p>{escape(value)}</p></div>"
            )
    source_html = "".join(source_rows) or "<div class='source-row'><span>Source</span><p>No source line available.</p></div>"

    st.markdown(
        f"""
        <section class="issue-card">
            <div class="issue-card-header">
                <div>
                    <h3>{escape(str(row["Medication"]))}</h3>
                    <div class="action-label">{escape(action_label(row))}</div>
                </div>
                <span class="priority {priority_class}">{escape(str(row["Review Priority"]))}</span>
            </div>
            <div class="issue-pills">{category_html}</div>
            <div class="details">{escape(str(row["Details"]))}</div>
            <div class="sources">{source_html}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_worklist_group(title: str, rows: pd.DataFrame, empty_text: str) -> None:
    st.markdown(f"#### {title}")
    if rows.empty:
        st.caption(empty_text)
        return

    for _, row in rows.iterrows():
        render_issue_card(row)


st.set_page_config(
    page_title="CPRS Medication Transition Assistant",
    page_icon="meds",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1380px;
        padding-top: 1.6rem;
    }
    div[data-testid="stTextArea"] textarea {
        font-size: 0.95rem;
        line-height: 1.35;
    }
    .issue-card {
        border: 1px solid #d8dee8;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 14px;
        background: #ffffff;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    }
    .issue-card-header {
        align-items: center;
        display: flex;
        gap: 12px;
        justify-content: space-between;
        margin-bottom: 8px;
    }
    .issue-card h3 {
        font-size: 1.08rem;
        line-height: 1.25;
        margin: 0;
    }
    .action-label {
        color: #475569;
        font-size: 0.86rem;
        font-weight: 650;
        margin-top: 4px;
    }
    .priority {
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        padding: 4px 10px;
        white-space: nowrap;
    }
    .priority-high {
        background: #fee2e2;
        color: #991b1b;
    }
    .priority-standard {
        background: #e0f2fe;
        color: #075985;
    }
    .issue-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-bottom: 10px;
    }
    .issue-pill {
        background: #eef2ff;
        border: 1px solid #c7d2fe;
        border-radius: 999px;
        color: #3730a3;
        font-size: 0.78rem;
        font-weight: 650;
        padding: 4px 9px;
    }
    .details {
        color: #334155;
        font-size: 0.92rem;
        line-height: 1.4;
        margin-bottom: 12px;
    }
    .sources {
        border-top: 1px solid #e5e7eb;
        padding-top: 10px;
    }
    .source-row {
        display: grid;
        gap: 10px;
        grid-template-columns: 92px 1fr;
        margin-top: 6px;
    }
    .source-row span {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .source-row p {
        color: #111827;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
        font-size: 0.86rem;
        line-height: 1.35;
        margin: 0;
        overflow-wrap: anywhere;
    }
    .timeline-wrap {
        border: 1px solid #d8dee8;
        border-radius: 8px;
        overflow-x: auto;
        background: #ffffff;
    }
    .timeline-table {
        border-collapse: collapse;
        min-width: 920px;
        width: 100%;
    }
    .timeline-table th {
        background: #f8fafc;
        border-bottom: 1px solid #d8dee8;
        color: #475569;
        font-size: 0.78rem;
        letter-spacing: 0;
        padding: 10px 12px;
        text-align: left;
        text-transform: uppercase;
    }
    .timeline-table td {
        border-bottom: 1px solid #edf2f7;
        color: #111827;
        font-size: 0.9rem;
        padding: 11px 12px;
        vertical-align: middle;
    }
    .timeline-table tr:last-child td {
        border-bottom: 0;
    }
    .timeline-med {
        font-weight: 700;
        min-width: 220px;
    }
    .status-pill,
    .transition-pill {
        border-radius: 999px;
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 4px 9px;
        white-space: nowrap;
    }
    .status-present {
        background: #dcfce7;
        color: #166534;
    }
    .status-absent {
        background: #f1f5f9;
        color: #64748b;
    }
    .transition-high {
        background: #fee2e2;
        color: #991b1b;
    }
    .transition-review {
        background: #fef3c7;
        color: #92400e;
    }
    .transition-change {
        background: #e0e7ff;
        color: #3730a3;
    }
    .transition-ok {
        background: #e0f2fe;
        color: #075985;
    }
    .board-legend {
        align-items: center;
        color: #475569;
        display: flex;
        flex-wrap: wrap;
        font-size: 0.86rem;
        gap: 16px;
        margin: 2px 0 12px;
    }
    .legend-dot {
        border-radius: 999px;
        display: inline-block;
        height: 10px;
        margin-right: 4px;
        width: 10px;
    }
    .legend-dot-on {
        background: #16a34a;
    }
    .legend-dot-off {
        background: #cbd5e1;
    }
    .recon-board {
        background: #ffffff;
        border: 1px solid #d8dee8;
        border-radius: 8px;
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1fr) minmax(190px, 0.7fr);
        overflow: hidden;
    }
    .recon-header {
        background: #f8fafc;
        border-bottom: 1px solid #d8dee8;
        color: #334155;
        font-size: 0.82rem;
        font-weight: 850;
        padding: 9px 10px;
        text-transform: uppercase;
    }
    .recon-review-header {
        color: #475569;
    }
    .recon-row {
        display: contents;
    }
    .recon-cell,
    .recon-review {
        border-bottom: 1px solid #edf2f7;
        min-height: 42px;
        padding: 8px 10px;
    }
    .recon-cell {
        align-items: flex-start;
        display: flex;
        flex-direction: column;
        font-size: 0.88rem;
        font-weight: 750;
        gap: 2px;
        justify-content: center;
        line-height: 1.2;
        overflow-wrap: anywhere;
    }
    .recon-med {
        display: block;
    }
    .recon-dose {
        color: #475569;
        display: block;
        font-size: 0.74rem;
        font-weight: 750;
        line-height: 1.15;
    }
    .recon-empty {
        display: block;
        width: 100%;
    }
    .recon-present {
        background: #f0fdf4;
        color: #14532d;
    }
    .recon-absent {
        background: #f8fafc;
        color: #cbd5e1;
        font-weight: 700;
        justify-content: center;
    }
    .recon-row-high .recon-cell,
    .recon-row-high .recon-review {
        box-shadow: inset 3px 0 0 #ef4444;
    }
    .recon-review {
        align-items: flex-start;
        display: flex;
        flex-direction: column;
        gap: 4px;
        justify-content: center;
    }
    .recon-reason {
        color: #64748b;
        font-size: 0.72rem;
        line-height: 1.2;
    }
    .board-action {
        border-radius: 999px;
        flex: 0 0 auto;
        font-size: 0.7rem;
        font-weight: 800;
        max-width: 160px;
        overflow: hidden;
        padding: 4px 8px;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .board-action-high {
        background: #fee2e2;
        color: #991b1b;
    }
    @media (max-width: 920px) {
        .recon-board {
            grid-template-columns: minmax(0, 1fr);
        }
        .recon-header {
            display: none;
        }
        .recon-row {
            border-bottom: 1px solid #d8dee8;
            display: grid;
            grid-template-columns: 1fr;
        }
        .recon-cell::before {
            color: #64748b;
            content: attr(data-label);
            font-size: 0.72rem;
            font-weight: 850;
            margin-right: 8px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Discharge Medication Review")
st.warning(
    "Do not enter PHI or patient identifiers."
)
st.caption(
    "Paste de-identified CPRS medication sections. Verify all findings against CPRS before signing orders."
)

if "example_loaded" not in st.session_state:
    st.session_state.example_loaded = False
if "review_ready" not in st.session_state:
    st.session_state.review_ready = False

left, right = st.columns([1, 1])
with left:
    if st.button("Load de-identified example"):
        admission_example, inpatient_example, discharge_example, notes_example = load_example()
        st.session_state.admission_text = admission_example
        st.session_state.inpatient_text = inpatient_example
        st.session_state.discharge_text = discharge_example
        st.session_state.notes_text = notes_example
        st.session_state.example_loaded = True
        st.session_state.review_ready = False

with right:
    if st.button("Clear all"):
        st.session_state.admission_text = ""
        st.session_state.inpatient_text = ""
        st.session_state.discharge_text = ""
        st.session_state.notes_text = ""
        st.session_state.review_ready = False
        st.session_state.llm_review = ""
        st.session_state.llm_answers = []

input_panel = st.expander("Edit CPRS input", expanded=not st.session_state.review_ready)
with input_panel:
    input_col_1, input_col_2 = st.columns(2)
    with input_col_1:
        admission_text = st.text_area(
            "Admission Medication Reconciliation",
            key="admission_text",
            height=220,
            placeholder="Example: Lisinopril 10 mg daily",
        )
        discharge_text = st.text_area(
            "Discharge Medication Reconciliation",
            key="discharge_text",
            height=220,
            placeholder="Example: Lisinopril 20 mg daily",
        )

    with input_col_2:
        inpatient_text = st.text_area(
            "Inpatient Medication Orders",
            key="inpatient_text",
            height=220,
            placeholder="Example: Insulin glargine 10 units qhs",
        )
        notes_text = st.text_area(
            "Clinical Notes",
            key="notes_text",
            height=220,
            placeholder="Paste only de-identified clinical context relevant to medication review.",
        )

    analyze = st.button("Analyze med lists", type="primary", width="stretch")

if analyze:
    if not any([admission_text.strip(), inpatient_text.strip(), discharge_text.strip()]):
        st.error("Enter at least one medication list before analyzing.")
    else:
        analysis = build_analysis(admission_text, inpatient_text, discharge_text, notes_text)
        timeline = build_timeline(admission_text, inpatient_text, discharge_text, analysis)
        summary = summarize_categories(analysis)
        discharge_available = has_discharge_list(discharge_text)
        st.session_state.analysis = analysis
        st.session_state.timeline = timeline
        st.session_state.summary = summary
        st.session_state.discharge_available = discharge_available
        st.session_state.payload = analysis_payload(analysis, timeline, summary)
        st.session_state.diagnostics = parser_diagnostics(admission_text, inpatient_text, discharge_text, timeline)
        st.session_state.signout = resident_signout(summary, analysis)
        st.session_state.review_ready = True

if st.session_state.review_ready:
    analysis = st.session_state.analysis
    timeline = st.session_state.timeline
    summary = st.session_state.summary
    payload = st.session_state.payload
    diagnostics = st.session_state.diagnostics
    discharge_available = st.session_state.discharge_available

    st.subheader("Discharge Med Rec Worklist")

    if analysis.empty:
        st.success("No medication transition issues were detected by the local parser.")
    else:
        metric_cols = st.columns(5 if not discharge_available else 6)
        metric_cols[0].metric("High Risk", summary["High Risk"])
        if discharge_available:
            metric_cols[1].metric("New at Discharge", summary["New"])
            metric_cols[2].metric("Inpatient Only", summary["Inpatient Only"])
            metric_cols[3].metric("Home Meds Stopped", summary["Discontinued"])
            metric_cols[4].metric("Held on Admission", summary["Held"])
            metric_cols[5].metric("Dose/Freq Changes", summary["Dose"] + summary["Frequency"])
        else:
            metric_cols[1].metric("Held on Admission", summary["Held"])
            metric_cols[2].metric("New Inpatient", summary["New Inpatient"])
            metric_cols[3].metric("Dose/Freq Changes", summary["Dose"] + summary["Frequency"])
            metric_cols[4].metric("Discharge Med Rec", "Not provided")

        st.markdown("#### Medication Board")
        render_medication_board(timeline, discharge_available)

        with st.expander("Detailed review", expanded=False):
            worklist_tab, timeline_tab, signout_tab, llm_tab, parser_tab, table_tab = st.tabs(
                ["Worklist", "Timeline", "Signout", "LLM Check", "Parser Check", "Raw Table"]
            )
            with worklist_tab:
                render_worklist_group(
                    "Check before discharge",
                    analysis[analysis["Review Priority"] == "High"],
                    "No high-risk medications flagged.",
                )
                render_worklist_group(
                    "Held on admission",
                    rows_with_category(analysis, "Held on admission", include_high_risk=False),
                    "No home medications held on admission flagged.",
                )
                render_worklist_group(
                    "New inpatient meds",
                    rows_with_category(analysis, "New inpatient med", include_high_risk=False),
                    "No new inpatient medications flagged.",
                )
                render_worklist_group(
                    "New discharge meds",
                    rows_with_category(analysis, "New at discharge", include_high_risk=False),
                    "No new discharge medications flagged.",
                )
                render_worklist_group(
                    "Home meds stopped",
                    rows_with_category(analysis, "Home med stopped", include_high_risk=False),
                    "No stopped home medications flagged.",
                )
                render_worklist_group(
                    "Inpatient meds not continued",
                    rows_with_category(analysis, "Inpatient med not continued", include_high_risk=False),
                    "No inpatient-only medications flagged.",
                )
                render_worklist_group(
                    "Dose or frequency changes",
                    analysis[
                        analysis["Category"].str.contains("Dose change|Frequency change", case=False, regex=True)
                        & (analysis["Review Priority"] != "High")
                    ],
                    "No dose or frequency changes flagged.",
                )

            with timeline_tab:
                render_timeline(timeline)

            with signout_tab:
                st.text_area(
                    "Copy resident signout",
                    value=st.session_state.signout,
                    height=220,
                )

            with llm_tab:
                render_llm_review(payload)

            with parser_tab:
                st.metric("Parser confidence", diagnostics["confidence"])
                st.table(
                    pd.DataFrame(
                        [
                            {"List": name, "Medications parsed": count}
                            for name, count in diagnostics["parsed_counts"].items()
                        ]
                    )
                )
                if diagnostics["suspect_names"]:
                    st.warning("These medication names may need a manual glance:")
                    st.write(", ".join(diagnostics["suspect_names"]))
                else:
                    st.success("No obviously awkward parsed medication names detected.")

            with table_tab:
                st.dataframe(
                    analysis,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Medication": st.column_config.TextColumn("Medication", width="medium"),
                        "Category": st.column_config.TextColumn("Issues", width="medium"),
                        "Admission": st.column_config.TextColumn("Admission", width="large"),
                        "Inpatient": st.column_config.TextColumn("Inpatient", width="large"),
                        "Discharge": st.column_config.TextColumn("Discharge", width="large"),
                        "Details": st.column_config.TextColumn("Details", width="large"),
                        "Review Priority": st.column_config.TextColumn("Priority", width="small"),
                    },
                )

        board_csv = board_export_dataframe(timeline, discharge_available).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download board as CSV",
            data=board_csv,
            file_name="medication_transition_review.csv",
            mime="text/csv",
        )

st.divider()
st.caption(
    "Clinical decision support only. Verify medications, indications, renal function, allergies, and discharge orders in CPRS."
)
