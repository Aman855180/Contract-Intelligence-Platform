"""
generate_synthetic_contracts.py

Generates 20 synthetic PDF contracts with deliberately planted
inconsistencies and known risk flags, along with a ground truth JSON file
used by evaluation/eval_harness.py.

Contract layout (3 pages per contract, known page numbers for Q&A benchmark):
  Page 1 — Parties, Services, Payment Terms, Effective Date
  Page 2 — Termination (if present), Confidentiality (if present), Liability
  Page 3 — Governing Law, Signature block (if present)

20 contracts in 2 groups:
  Group 1 (c01-c10): Comparison pairs for inconsistency detection evaluation
    - c01/c02: payment term mismatch (30 vs 60 days)         → planted TP
    - c03/c04: payment term mismatch (30 vs 90 days)         → planted TP
    - c05/c06: payment term mismatch (14 vs 45 days)         → planted TP
    - c07/c08: effective date mismatch (Jan vs Mar 2025)     → planted TP
    - c09/c10: no inconsistency (identical terms)            → TN baseline

  Group 2 (c11-c20): Risk flag benchmark
    - c11-c14: all clauses present (no rule flags expected)
    - c15:     missing signature block
    - c16:     missing termination clause
    - c17:     missing confidentiality clause
    - c18:     missing signature + termination
    - c19:     missing confidentiality + signature
    - c20:     all three clauses missing (maximum rule-based risk)

Q&A benchmark: 20 questions across 10 contracts (2 per contract),
with known source pages for BM25 retrieval recall measurement.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OUTPUT_DIR = _REPO_ROOT / "evaluation" / "test_contracts"
_GROUND_TRUTH_PATH = _REPO_ROOT / "evaluation" / "ground_truth.json"


# ---------------------------------------------------------------------------
# Contract spec dataclass
# ---------------------------------------------------------------------------

@dataclass
class ContractSpec:
    contract_id: str
    party_a: str
    party_b: str
    effective_date: str
    payment_days: int
    notice_days: int
    liability_cap: str
    has_termination: bool = True
    has_confidentiality: bool = True
    has_signature: bool = True
    services_description: str = "professional software consulting"


# ---------------------------------------------------------------------------
# Page content builders (deterministic so page numbers are predictable)
# ---------------------------------------------------------------------------

def _page1(spec: ContractSpec) -> str:
    return f"""SERVICE AGREEMENT

This Service Agreement ("Agreement") is entered into as of {spec.effective_date}
by and between {spec.party_a} ("Client") and {spec.party_b} ("Vendor").

1. SERVICES
Vendor shall provide {spec.services_description} services to Client as described
in any applicable Statement of Work executed by both parties. Vendor shall
perform the services in a professional and workmanlike manner.

2. PAYMENT TERMS
Client shall pay all undisputed invoices within {spec.payment_days} days of
receipt of each invoice from Vendor. Payment shall be made by electronic wire
transfer to the bank account designated in writing by Vendor. All amounts are
stated in United States Dollars. Invoices not disputed within ten (10) days of
receipt shall be deemed accepted. Late payments shall accrue interest at 1.5
percent per month from the due date until paid in full. All fees paid are
non-refundable once Vendor has commenced delivery of the applicable services.

3. EFFECTIVE DATE AND TERM
This Agreement is effective as of {spec.effective_date} and shall continue for
an initial term of one (1) year unless earlier terminated in accordance with
the provisions of this Agreement."""


def _page2(spec: ContractSpec) -> str:
    termination = f"""4. TERMINATION
Either party may terminate this Agreement for any reason upon {spec.notice_days}
days' prior written notice to the other party. In the event of a material breach
by either party, the non-breaching party may terminate immediately upon written
notice if the breach is not cured within fifteen (15) days after written notice
of the breach. Upon termination, Client shall pay Vendor for all services
rendered through the date of termination.""" if spec.has_termination else \
    """4. DURATION
This Agreement shall remain in effect for a fixed period of one (1) year from
the effective date. Upon expiry of this period, the Agreement may only be
extended by mutual written consent signed by both parties. Neither party has
a unilateral right of early exit under this Agreement."""

    confidentiality = """5. CONFIDENTIALITY
Each party agrees to keep strictly confidential all non-public, proprietary,
or confidential information received from the other party ("Confidential
Information") and shall not disclose such Confidential Information to any
third party without the prior written consent of the disclosing party.
Each party shall use Confidential Information solely for the purposes of
this Agreement. The obligations of this Section shall survive termination.""" if spec.has_confidentiality else \
    """5. INFORMATION SHARING
The parties acknowledge that information may be shared in the course of
performing services. Each party shall use reasonable care to protect any
sensitive business information shared by the other party."""

    return f"""{termination}

{confidentiality}

6. LIMITATION OF LIABILITY
Each party's total aggregate liability arising out of or related to this
Agreement shall be limited to {spec.liability_cap}. In no event shall either
party be liable for any indirect, incidental, special, punitive, or
consequential damages, regardless of the cause of action or the theory of
liability, even if such party has been advised of the possibility of such
damages. This limitation shall not apply to indemnification obligations or
payment obligations expressly stated in this Agreement."""


def _page3(spec: ContractSpec) -> str:
    signature_block = f"""IN WITNESS WHEREOF, the authorized representatives of the parties have
executed this Agreement as of {spec.effective_date}.

_________________________________       _________________________________
{spec.party_a}                         {spec.party_b}
Authorized Representative               Authorized Representative
Date: _________________________         Date: _________________________""" \
    if spec.has_signature else \
    """This Agreement becomes effective upon the mutual written confirmation of
both parties via electronic mail. Physical execution is not required under
the terms of this Agreement. Each party's acceptance via email constitutes
full agreement to all terms and conditions set forth herein."""

    return f"""7. GOVERNING LAW AND DISPUTES
This Agreement shall be governed by and construed in accordance with the laws
of the State of Delaware, without regard to its conflict of law provisions.
Any dispute arising under this Agreement shall first be submitted to mediation
before either party may commence litigation.

8. ENTIRE AGREEMENT
This Agreement constitutes the entire agreement between the parties with
respect to the subject matter hereof and supersedes all prior agreements,
understandings, negotiations, and discussions between the parties.

9. AMENDMENTS
No amendment or modification of this Agreement shall be binding unless made in
writing and agreed to by authorized representatives of both parties.

{signature_block}"""


# ---------------------------------------------------------------------------
# PDF writer
# ---------------------------------------------------------------------------

def _write_pdf(path: Path, spec: ContractSpec) -> None:
    doc = fitz.open()
    rect = fitz.Rect(72, 72, 540, 750)
    for content in [_page1(spec), _page2(spec), _page3(spec)]:
        page = doc.new_page()
        page.insert_textbox(rect, content, fontsize=10, fontname="helv")
    doc.save(str(path))
    doc.close()


# ---------------------------------------------------------------------------
# Contract specs
# ---------------------------------------------------------------------------

def _specs() -> list[ContractSpec]:
    return [
        # ── Group 1: Comparison pairs ────────────────────────────────────────
        # Pair 01-02: payment 30 vs 60 days
        ContractSpec("c01", "Acme Corp", "Vertex Solutions", "January 1, 2025", 30, 60, "total fees paid"),
        ContractSpec("c02", "Acme Corp", "Vertex Solutions", "January 1, 2025", 60, 60, "total fees paid"),
        # Pair 03-04: payment 30 vs 90 days
        ContractSpec("c03", "Nexus Inc", "Orbital Tech", "February 1, 2025", 30, 30, "fees paid in prior quarter"),
        ContractSpec("c04", "Nexus Inc", "Orbital Tech", "February 1, 2025", 90, 30, "fees paid in prior quarter"),
        # Pair 05-06: payment 14 vs 45 days
        ContractSpec("c05", "Pinnacle Ltd", "Stratum Group", "March 1, 2025", 14, 60, "two hundred thousand dollars"),
        ContractSpec("c06", "Pinnacle Ltd", "Stratum Group", "March 1, 2025", 45, 60, "two hundred thousand dollars"),
        # Pair 07-08: date mismatch (January vs March 2025)
        ContractSpec("c07", "Helios Partners", "Zenith LLC", "January 1, 2025", 30, 60, "total annual fees"),
        ContractSpec("c08", "Helios Partners", "Zenith LLC", "March 1, 2025", 30, 60, "total annual fees"),
        # Pair 09-10: no inconsistency (true negative baseline)
        ContractSpec("c09", "Apex Corp", "Delta Services", "April 1, 2025", 30, 60, "one hundred thousand dollars"),
        ContractSpec("c10", "Apex Corp", "Delta Services", "April 1, 2025", 30, 60, "one hundred thousand dollars"),

        # ── Group 2: Risk flag benchmark ─────────────────────────────────────
        ContractSpec("c11", "Atlas Inc", "Mercury Tech", "January 15, 2025", 30, 60, "total fees"),
        ContractSpec("c12", "Orion Ltd", "Sirius Group", "February 15, 2025", 45, 30, "fees in prior year"),
        ContractSpec("c13", "Solaris Corp", "Vega Solutions", "March 15, 2025", 60, 90, "one million dollars"),
        ContractSpec("c14", "Polaris Inc", "Andromeda LLC", "April 15, 2025", 30, 60, "total contract value"),
        # c15: missing signature
        ContractSpec("c15", "Rigel Corp", "Lyra Tech", "May 1, 2025", 30, 60, "total fees", has_signature=False),
        # c16: missing termination clause
        ContractSpec("c16", "Cygnus Inc", "Aquila LLC", "June 1, 2025", 30, 60, "total fees", has_termination=False),
        # c17: missing confidentiality clause
        ContractSpec("c17", "Vela Corp", "Libra Tech", "July 1, 2025", 30, 60, "total fees", has_confidentiality=False),
        # c18: missing signature + termination
        ContractSpec("c18", "Corvus Ltd", "Lupus Group", "August 1, 2025", 30, 60, "total fees",
                     has_signature=False, has_termination=False),
        # c19: missing confidentiality + signature
        ContractSpec("c19", "Pavo Inc", "Tucana LLC", "September 1, 2025", 30, 60, "total fees",
                     has_confidentiality=False, has_signature=False),
        # c20: all missing (maximum risk)
        ContractSpec("c20", "Grus Corp", "Indus Tech", "October 1, 2025", 30, 60, "total fees",
                     has_signature=False, has_termination=False, has_confidentiality=False),
    ]


# ---------------------------------------------------------------------------
# Ground truth builder
# ---------------------------------------------------------------------------

def _build_ground_truth(specs: list[ContractSpec]) -> dict:
    spec_map = {s.contract_id: s for s in specs}

    contracts = {}
    for s in specs:
        contracts[s.contract_id] = {
            "filename": f"{s.contract_id}_contract.pdf",
            "party_a": s.party_a,
            "party_b": s.party_b,
            "effective_date": s.effective_date,
            "payment_days": s.payment_days,
            "notice_days": s.notice_days,
            "has_termination": s.has_termination,
            "has_confidentiality": s.has_confidentiality,
            "has_signature": s.has_signature,
            "expected_rule_risk_flags": _expected_risk_flags(s),
        }

    comparison_pairs = [
        # Planted TP: numeric payment mismatch
        _pair("pair_01_payment", "c01", "c02", [
            {"field_type": "payment_terms", "method": "rule_based_numeric_date",
             "description": f"payment {spec_map['c01'].payment_days} vs {spec_map['c02'].payment_days} days"}
        ]),
        _pair("pair_02_payment", "c03", "c04", [
            {"field_type": "payment_terms", "method": "rule_based_numeric_date",
             "description": f"payment {spec_map['c03'].payment_days} vs {spec_map['c04'].payment_days} days"}
        ]),
        _pair("pair_03_payment", "c05", "c06", [
            {"field_type": "payment_terms", "method": "rule_based_numeric_date",
             "description": f"payment {spec_map['c05'].payment_days} vs {spec_map['c06'].payment_days} days"}
        ]),
        # Planted TP: date mismatch
        _pair("pair_04_date", "c07", "c08", [
            {"field_type": "key_dates", "method": "rule_based_numeric_date",
             "description": "effective date January vs March 2025"}
        ]),
        # TN: no inconsistency
        _pair("pair_05_clean", "c09", "c10", []),
    ]

    # Q&A benchmark: 2 questions per contract, 10 contracts
    qa_benchmark = []
    qa_contracts = ["c01", "c03", "c05", "c07", "c09", "c11", "c13", "c15", "c17", "c19"]
    for cid in qa_contracts:
        s = spec_map[cid]
        qa_benchmark.append({
            "qa_id": f"qa_{cid}_payment",
            "contract_id": cid,
            "question": "What is the payment term?",
            "expected_keywords": [str(s.payment_days), "days"],
            "source_page": 1,
        })
        if s.has_termination:
            qa_benchmark.append({
                "qa_id": f"qa_{cid}_termination",
                "contract_id": cid,
                "question": "What is the notice period for termination?",
                "expected_keywords": [str(s.notice_days), "days"],
                "source_page": 2,
            })
        else:
            qa_benchmark.append({
                "qa_id": f"qa_{cid}_liability",
                "contract_id": cid,
                "question": "What is the liability limitation?",
                "expected_keywords": ["liability", "limited"],
                "source_page": 2,
            })

    return {
        "version": "1.0",
        "total_contracts": len(specs),
        "contracts": contracts,
        "comparison_pairs": comparison_pairs,
        "qa_benchmark": qa_benchmark,
        "risk_benchmark": [
            {"contract_id": cid, "expected_triggered_rule_flags": contracts[cid]["expected_rule_risk_flags"]}
            for cid in contracts
        ],
    }


def _expected_risk_flags(s: ContractSpec) -> list[str]:
    flags = []
    if not s.has_signature:
        flags.append("missing_signature_field")
    if not s.has_termination:
        flags.append("missing_termination_clause")
    if not s.has_confidentiality:
        flags.append("missing_confidentiality_clause")
    return flags


def _pair(pair_id: str, a: str, b: str, inconsistencies: list[dict]) -> dict:
    return {
        "pair_id": pair_id,
        "contract_a_id": a,
        "contract_b_id": b,
        "expected_inconsistencies": inconsistencies,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate(output_dir: Path = _OUTPUT_DIR, ground_truth_path: Path = _GROUND_TRUTH_PATH) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    specs = _specs()
    generated = []
    for spec in specs:
        pdf_path = output_dir / f"{spec.contract_id}_contract.pdf"
        _write_pdf(pdf_path, spec)
        generated.append(pdf_path.name)
        print(f"  Generated {pdf_path.name}")

    gt = _build_ground_truth(specs)
    ground_truth_path.write_text(json.dumps(gt, indent=2))
    print(f"\nGround truth saved to {ground_truth_path}")
    print(f"Generated {len(generated)} contracts in {output_dir}")


if __name__ == "__main__":
    print("Generating synthetic contracts...")
    generate()
