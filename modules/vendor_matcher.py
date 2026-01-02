"""
Vendor Matcher Module
Uses fuzzy logic to normalize supplier names based on configured aliases.
"""
import logging
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Any

class VendorMatcher:
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = False
        self.mappings: Dict[str, str] = {}
        
        # Load from config
        alias_conf = config.get("postbatch", {}).get("vendor_aliases", {})
        if alias_conf.get("enabled", False):
            self.enabled = True
            raw_mappings = alias_conf.get("mappings", {})
            self._build_lookup_table(raw_mappings)

    def _build_lookup_table(self, raw_mappings: Dict[str, List[str]]):
        """
        Inverts the mapping config:
        From: { "Amazon": ["Amzn", "AWS"] }
        To: { "amzn": "Amazon", "aws": "Amazon", "amazon": "Amazon" }
        """
        for canonical, aliases in raw_mappings.items():
            # Add canonical itself
            self.mappings[canonical.lower()] = canonical
            # Add all aliases
            if isinstance(aliases, list):
                for alias in aliases:
                    self.mappings[alias.lower()] = canonical

    def normalize(self, extracted_name: Optional[str]) -> Optional[str]:
        """
        Returns the canonical name if a match is found, otherwise returns original.
        """
        if not self.enabled or not extracted_name:
            return extracted_name

        clean_name = extracted_name.strip()
        lower_name = clean_name.lower()

        # 1. Exact match (case insensitive)
        if lower_name in self.mappings:
            canonical = self.mappings[lower_name]
            self.logger.info(f"🧩 Exact Alias Match: '{extracted_name}' -> '{canonical}'")
            return canonical

        # 2. Fuzzy match
        # We look for the best match among keys in self.mappings
        best_match = None
        best_ratio = 0.0
        
        for alias, canonical in self.mappings.items():
            ratio = SequenceMatcher(None, lower_name, alias).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = canonical

        # Threshold for fuzzy match (0.85 is a safe bet for slight typos)
        if best_ratio > 0.85:
            self.logger.info(f"🧩 Fuzzy Alias Match ({best_ratio:.2f}): '{extracted_name}' -> '{best_match}'")
            return best_match

        return extracted_name
