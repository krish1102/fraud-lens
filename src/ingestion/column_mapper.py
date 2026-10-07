"""
Column Mapper: Automatically maps user CSV columns to canonical schema with confidence scores and fuzzy fallback.
"""

from typing import Dict, List, Any, Optional
import difflib
from src.configs.settings import CANONICAL_FIELDS, COLUMN_ALIASES


class ColumnMapper:
    """Maps arbitrary user CSV columns to the FraudLens canonical schema."""

    def __init__(self, aliases: Optional[Dict[str, List[str]]] = None):
        self.aliases = aliases or COLUMN_ALIASES

    def suggest_mappings(self, source_columns: List[str]) -> Dict[str, Any]:
        """
        Takes raw source columns and suggests canonical field mappings.
        Returns:
            dict with:
                - 'mapped': {canonical_field: source_column}
                - 'details': {canonical_field: {'source_column': ..., 'confidence': ..., 'method': ...}}
                - 'unmapped_source': list of columns not mapped
                - 'missing_required': list of required canonical fields not found
                - 'ambiguous': list of canonical fields with multiple potential candidate matches
        """
        mapped: Dict[str, str] = {}
        details: Dict[str, Dict[str, Any]] = {}
        used_sources = set()
        ambiguous_candidates: Dict[str, List[Dict[str, Any]]] = {}

        # Normalize source names for comparison: strip and lowercase
        norm_source_map = {col: col.strip().lower().replace(" ", "_").replace("-", "_") for col in source_columns}

        # Step 1 & 2: Exact canonical match & alias list match
        for canonical, info in CANONICAL_FIELDS.items():
            candidates = []
            canonical_aliases = [canonical] + self.aliases.get(canonical, [])

            for src_orig, src_norm in norm_source_map.items():
                if src_norm == canonical:
                    candidates.append({"source": src_orig, "confidence": 1.0, "method": "exact_canonical"})
                elif src_norm in canonical_aliases:
                    candidates.append({"source": src_orig, "confidence": 0.95, "method": "known_alias"})
                else:
                    # Check substring match (e.g. 'amount_usd' contains 'amount')
                    for alias in canonical_aliases:
                        if alias in src_norm or src_norm in alias:
                            ratio = difflib.SequenceMatcher(None, src_norm, alias).ratio()
                            if ratio >= 0.70:
                                candidates.append({
                                    "source": src_orig,
                                    "confidence": round(0.70 + 0.20 * ratio, 2),
                                    "method": "fuzzy_substring"
                                })
                                break

            # Sort candidate matches by confidence descending
            candidates = sorted(candidates, key=lambda c: c["confidence"], reverse=True)

            if len(candidates) > 1 and candidates[0]["confidence"] == candidates[1]["confidence"]:
                ambiguous_candidates[canonical] = candidates
            elif len(candidates) >= 1:
                best = candidates[0]
                if best["source"] not in used_sources:
                    mapped[canonical] = best["source"]
                    details[canonical] = best
                    used_sources.add(best["source"])

        # Check required fields
        missing_required = [
            field for field, conf in CANONICAL_FIELDS.items()
            if conf["required"] and field not in mapped
        ]

        unmapped_source = [col for col in source_columns if col not in used_sources]

        return {
            "mapped": mapped,
            "details": details,
            "unmapped_source": unmapped_source,
            "missing_required": missing_required,
            "ambiguous": ambiguous_candidates
        }

    @staticmethod
    def apply_mapping(
        df,
        confirmed_mapping: Dict[str, str],
        drop_unmapped: bool = False
    ):
        """
        Renames confirmed source columns into canonical field names.
        """
        rename_dict = {src: canonical for canonical, src in confirmed_mapping.items() if src in df.columns}
        canonical_df = df.rename(columns=rename_dict)
        if drop_unmapped:
            keep_cols = [c for c in confirmed_mapping.keys() if c in canonical_df.columns]
            return canonical_df[keep_cols]
        return canonical_df
