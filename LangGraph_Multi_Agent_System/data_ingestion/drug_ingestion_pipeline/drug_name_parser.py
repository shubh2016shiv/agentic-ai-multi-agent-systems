"""
Drug Name Parser
==================
Parses STRUCTURED clinical fields from raw drug name strings.

WHY THIS EXISTS
---------------
The source Excel files contain drug names that are actually COMPOUND strings
embedding the dosage form, strength, and route of administration:

    "Pharbedryl Capsule 25 Mg Oral"
     ^^^^^^^^^  ^^^^^^^  ^^^^^  ^^^^
     Brand name Form     Strength Route

These structured fields — dosage_strength, dosage_form, route_of_administration —
are clinically important and needed for safety checks, substitution logic, and
clinical decision support. Leaving them buried in a string wastes them entirely.

PARSING STRATEGY
----------------
1. Route detection  — scan for known route keywords at the END of the name.
2. Form detection   — scan for known dosage form keywords after removing route.
3. Strength parsing — regex scan for numeric patterns (e.g., "25 Mg", "0.5 Mcg/Ml").

This is a BEST-EFFORT parser. It will miss unusual formats and that is acceptable.
The LLM enrichment step then fills any remaining gaps in route_of_administration
and dosage_forms via its own knowledge.

EXAMPLES
--------
    "Pharbedryl Capsule 25 Mg Oral"
        → form="Capsule", strength="25 Mg", route="Oral"

    "Metoprolol Succinate Tablet 50 Mg"
        → form="Tablet", strength="50 Mg", route=None

    "Amoxicillin Oral Suspension 250 Mg/5Ml"
        → form="Oral Suspension", strength="250 Mg/5Ml", route="Oral"

    "Lidocaine HCl 2% Injection Solution"
        → form="Injection Solution", strength="2%", route=None

    "Timolol Ophthalmic Solution 0.5%"
        → form="Ophthalmic Solution", strength="0.5%", route="Ophthalmic"

Usage:
    from data_ingestion.drug_ingestion_pipeline.drug_name_parser import DrugNameParser

    parsed = DrugNameParser.parse("Pharbedryl Capsule 25 Mg Oral")
    print(parsed.dosage_form)     # "Capsule"
    print(parsed.dosage_strength) # "25 Mg"
    print(parsed.parsed_route)    # "Oral"
"""

import re
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# KNOWN VOCABULARY FOR EACH PARSED FIELD
# =============================================================================
# Ordered from most specific (multi-word) to least specific (single word)
# so the longest match wins when scanning.
# =============================================================================

# Routes of administration — checked at word boundaries, case-insensitive.
# Order matters: longer phrases first to prevent partial matches.
_KNOWN_ROUTES_ORDERED_LONGEST_FIRST: list[str] = [
    "intravenous",
    "intramuscular",
    "subcutaneous",
    "intrathecal",
    "intradermal",
    "intravitreal",
    "intraperitoneal",
    "intraarticular",
    "intranasal",
    "intravaginal",
    "intrarectal",
    "transdermal",
    "transmucosal",
    "sublingual",
    "buccal",
    "ophthalmic",
    "otic",
    "rectal",
    "vaginal",
    "topical",
    "inhalation",
    "inhaled",
    "oral",
    "iv",
    "im",
    "sc",
    "sq",
    "po",
    "pr",
]

# Dosage forms — ordered longest-first to prefer "Extended Release Tablet"
# over just "Tablet".
_KNOWN_DOSAGE_FORMS_ORDERED_LONGEST_FIRST: list[str] = [
    "extended release tablet",
    "extended-release tablet",
    "extended release capsule",
    "extended-release capsule",
    "delayed release tablet",
    "delayed-release tablet",
    "delayed release capsule",
    "delayed-release capsule",
    "controlled release tablet",
    "sustained release tablet",
    "oral disintegrating tablet",
    "orally disintegrating tablet",
    "chewable tablet",
    "effervescent tablet",
    "film coated tablet",
    "enteric coated tablet",
    "transdermal patch",
    "transdermal gel",
    "topical cream",
    "topical gel",
    "topical lotion",
    "topical ointment",
    "topical solution",
    "ophthalmic solution",
    "ophthalmic ointment",
    "ophthalmic suspension",
    "ophthalmic gel",
    "otic solution",
    "otic suspension",
    "nasal spray",
    "nasal solution",
    "nasal suspension",
    "inhalation solution",
    "inhalation powder",
    "inhalation suspension",
    "inhalation aerosol",
    "metered dose inhaler",
    "dry powder inhaler",
    "oral solution",
    "oral suspension",
    "oral liquid",
    "oral drops",
    "oral syrup",
    "injection solution",
    "injection suspension",
    "for injection",
    "injectable solution",
    "injectable suspension",
    "prefilled syringe",
    "pre-filled syringe",
    "lyophilized powder",
    "powder for injection",
    "rectal suppository",
    "vaginal suppository",
    "vaginal cream",
    "vaginal gel",
    "vaginal ring",
    "buccal tablet",
    "buccal film",
    "sublingual tablet",
    "sublingual film",
    "sublingual spray",
    "lozenge",
    "troche",
    "gum",
    "mouthwash",
    "gargle",
    "enema",
    "suppository",
    "capsule",
    "tablet",
    "solution",
    "suspension",
    "syrup",
    "elixir",
    "emulsion",
    "lotion",
    "cream",
    "ointment",
    "gel",
    "foam",
    "spray",
    "aerosol",
    "powder",
    "granule",
    "patch",
    "film",
    "implant",
    "pellet",
    "injection",
    "infusion",
]

# Dosage strength patterns — matches things like:
#   "25 Mg", "0.5 Mcg", "500 Mg/5Ml", "1000 Units/Ml", "2%", "0.1%"
_DOSAGE_STRENGTH_REGEX = re.compile(
    r"""
    (?:                             # Numeric + unit pattern
        \d+(?:\.\d+)?               # Integer or decimal number (e.g., 25, 0.5)
        \s*                         # Optional whitespace
        (?:                         # Unit group
            mg|mcg|µg|ug|g|kg|ml|l  # Mass/volume
            |units?|iu              # Biological units
            |meq|mmol|mol           # Molar
            |%                      # Percentage (no space before %)
        )
        (?:                         # Optional per-unit (e.g., /5ml, /ml)
            \s*/\s*
            \d*\s*
            (?:mg|mcg|g|ml|l|units?)
        )?
    )
    |
    (?:\d+(?:\.\d+)?\s*%)          # Standalone percentage (e.g., "2%")
    """,
    re.VERBOSE | re.IGNORECASE,
)


# =============================================================================
# Parsed Result
# =============================================================================


@dataclass
class DrugNameParseResult:
    """
    Result of parsing a raw drug name string for structured clinical fields.

    All fields are Optional — the parser is best-effort and will return None
    for any field it cannot confidently identify.

    Attributes:
        dosage_form: Dosage form keyword extracted from the name
                     (e.g., "Capsule", "Oral Suspension", "Ophthalmic Solution").
        dosage_strength: Dosage strength string (e.g., "25 Mg", "250 Mg/5Ml", "2%").
        parsed_route: Route of administration detected in the name
                      (e.g., "Oral", "Intravenous", "Ophthalmic").
    """
    dosage_form: Optional[str]
    dosage_strength: Optional[str]
    parsed_route: Optional[str]


# =============================================================================
# Parser
# =============================================================================


class DrugNameParser:
    """
    Stateless parser that extracts structured pharmaceutical fields from
    raw drug name strings in a best-effort, non-destructive way.

    All methods are static — no instantiation needed.
    """

    @staticmethod
    def parse(raw_drug_name: str) -> DrugNameParseResult:
        """
        Parse dosage_form, dosage_strength, and route_of_administration
        from a raw drug name string.

        Args:
            raw_drug_name: The full drug name as it appears in the Excel file,
                           e.g., "Pharbedryl Capsule 25 Mg Oral".

        Returns:
            DrugNameParseResult with parsed fields (any/all may be None).
        """
        if not raw_drug_name or not raw_drug_name.strip():
            return DrugNameParseResult(
                dosage_form=None,
                dosage_strength=None,
                parsed_route=None,
            )

        lower_name = raw_drug_name.lower()

        parsed_route = DrugNameParser._detect_route(lower_name)
        dosage_form = DrugNameParser._detect_dosage_form(lower_name)
        dosage_strength = DrugNameParser._detect_dosage_strength(lower_name)

        return DrugNameParseResult(
            dosage_form=dosage_form,
            dosage_strength=dosage_strength,
            parsed_route=parsed_route,
        )

    @staticmethod
    def _detect_route(lower_name: str) -> Optional[str]:
        """
        Scan for known route-of-administration keywords in the drug name.

        Returns the FIRST matching route found (checking longest keywords first).
        The matched keyword is Title-Cased for display consistency.
        """
        for route_keyword in _KNOWN_ROUTES_ORDERED_LONGEST_FIRST:
            # Use word-boundary matching to avoid false positives like
            # "oral" inside "intravaginal"
            pattern = r"\b" + re.escape(route_keyword) + r"\b"
            if re.search(pattern, lower_name, re.IGNORECASE):
                # Map common abbreviations to readable labels
                abbreviation_map = {
                    "iv": "Intravenous",
                    "im": "Intramuscular",
                    "sc": "Subcutaneous",
                    "sq": "Subcutaneous",
                    "po": "Oral",
                    "pr": "Rectal",
                }
                canonical = abbreviation_map.get(route_keyword, route_keyword.title())
                return canonical
        return None

    @staticmethod
    def _detect_dosage_form(lower_name: str) -> Optional[str]:
        """
        Scan for known dosage form keywords in the drug name.

        Returns the LONGEST matching form phrase found (checking multi-word
        phrases before single-word phrases).
        """
        for form_phrase in _KNOWN_DOSAGE_FORMS_ORDERED_LONGEST_FIRST:
            pattern = r"\b" + re.escape(form_phrase) + r"\b"
            if re.search(pattern, lower_name, re.IGNORECASE):
                return form_phrase.title()
        return None

    @staticmethod
    def _detect_dosage_strength(lower_name: str) -> Optional[str]:
        """
        Scan for numeric dosage strength patterns (e.g., "25 Mg", "2%").

        Returns the FIRST strength match found, normalized with consistent
        spacing and capitalized unit (e.g., "25 Mg" not "25mg").
        """
        match = _DOSAGE_STRENGTH_REGEX.search(lower_name)
        if match:
            raw_strength = match.group(0).strip()
            # Normalize spacing around slash in compound strengths (500 Mg/5Ml)
            normalized = re.sub(r"\s*/\s*", "/", raw_strength)
            # Capitalize units (mg → Mg, mcg → Mcg, etc.)
            normalized = re.sub(
                r"(mg|mcg|µg|ug|g|kg|ml|l|units?|iu|meq|mmol|mol)",
                lambda m: m.group(0).capitalize(),
                normalized,
                flags=re.IGNORECASE,
            )
            return normalized.strip()
        return None
