"""
pii.py — PII (Personally Identifiable Information) detection and masking.

Supports detection and masking of:
  - Email addresses
  - Phone numbers (Indian + international)
  - Credit card numbers (16-digit)
  - Aadhaar-style numbers (12-digit)
"""

import re
import os
from typing import List, Dict

PII_PATTERNS: Dict[str, re.Pattern] = {
    "Email": re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    "Phone": re.compile(r"(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{2,5}\)?[\s\-]?)?\d{5,10}"),
    "CreditCard": re.compile(r"\b(?:\d[ \-]?){13,16}\d\b"),
    "Aadhaar": re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"),
}

PII_MASKS: Dict[str, str] = {
    "Email": "***@***.***",
    "Phone": "**********",
    "CreditCard": "****-****-****-****",
    "Aadhaar": "****-****-****",
}


def detect_pii(text: str) -> List[Dict[str, str]]:
    """Scan text for PII. Returns list of dicts with type, match, start, end."""
    findings = []
    for pii_type, pattern in PII_PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append({"type": pii_type, "match": match.group(), "start": match.start(), "end": match.end()})
    return findings


def mask_pii(text: str) -> str:
    """Replace all detected PII with corresponding masks."""
    masked = text
    for pii_type, pattern in PII_PATTERNS.items():
        masked = pattern.sub(PII_MASKS[pii_type], masked)
    return masked


def generate_privacy_report(text: str, output_path: str) -> str:
    """Scan text for PII and write a summary privacy report."""
    findings = detect_pii(text)
    type_counts: Dict[str, int] = {}
    for f in findings:
        type_counts[f["type"]] = type_counts.get(f["type"], 0) + 1

    lines = ["=" * 60, "          PRIVACY REPORT — PII SCAN RESULTS", "=" * 60, "",
             f"Total PII instances detected: {len(findings)}", ""]

    if type_counts:
        lines.append("Breakdown by PII Type:")
        lines.append("-" * 40)
        for pii_type, count in type_counts.items():
            lines.append(f"  {pii_type:15s} : {count}")
        lines.append("")

    lines.append("Detailed Findings:")
    lines.append("-" * 40)
    for i, f in enumerate(findings, 1):
        lines.append(f"  [{i}] Type={f['type']:<15s}  Match=\"{f['match']}\"  Pos={f['start']}-{f['end']}")
    if not findings:
        lines.append("  No PII detected.")

    lines += ["", "Masked Output:", "-" * 40, mask_pii(text), "", "=" * 60]
    report = "\n".join(lines)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    return report
