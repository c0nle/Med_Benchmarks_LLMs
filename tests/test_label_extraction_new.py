"""
Unit-Tests fuer label_extraction_mamma und label_extraction_arm.

Alle Tests verwenden ausschliesslich synthetische Mini-Daten – keine Patientendaten.
"""
import json
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===========================================================================
# Normalisierungs-Tests (Menopause, BIRADS, ACR, Laesionen)
# ===========================================================================

class TestNormalization:
    """Tests fuer _build_normalizer und _normalize_val / _normalize_birads / _normalize_acr."""

    def setup_method(self):
        from evaluate import _build_normalizer, _normalize_val, _normalize_birads, _normalize_acr
        self.build_normalizer = _build_normalizer
        self.normalize_val    = _normalize_val
        self.normalize_birads = _normalize_birads
        self.normalize_acr    = _normalize_acr

    def test_menopause_postmenopausal(self):
        norm = self.build_normalizer({"post": ["Postmenopausal", "postmenopause"]})
        assert self.normalize_val("Postmenopausal", norm) == "post"
        assert self.normalize_val("postmenopause",  norm) == "post"

    def test_menopause_praefix(self):
        norm = self.build_normalizer({"prä": ["prämenopausal", "prä"]})
        assert self.normalize_val("prämenopausal", norm) == "prä"
        assert self.normalize_val("PRÄMENOPAUSAL", norm) == "prä"

    def test_menopause_unknown_passthrough(self):
        norm = self.build_normalizer({"post": ["postmenopausal"]})
        # Unbekannter Wert → lowercase durchgereicht
        result = self.normalize_val("Sonstiges", norm)
        assert result == "sonstiges"

    def test_normalize_val_none(self):
        norm = self.build_normalizer({"post": ["postmenopausal"]})
        assert self.normalize_val(None, norm) is None
        assert self.normalize_val("",   norm) is None
        assert self.normalize_val("?",  norm) is None

    def test_birads_descriptive_text(self):
        norm = self.build_normalizer({"2": ["2 sicher benigne", "benigne"]})
        assert self.normalize_birads("2 Sicher benigne", norm) == "2"
        assert self.normalize_birads("2",                norm) == "2"

    def test_birads_leading_digit(self):
        norm = self.build_normalizer({})
        # Kein Mapping → fuehrende Ziffer extrahieren
        assert self.normalize_birads("4 Suspekt", norm) == "4"
        assert self.normalize_birads("5 Hochgradig suggestiv für Malignität", norm) == "5"

    def test_birads6_map_to_5(self):
        norm = self.build_normalizer({})
        assert self.normalize_birads("6 nachgewiesene Malignität", norm, "map_to_5") == "5"

    def test_birads6_keep(self):
        norm = self.build_normalizer({})
        assert self.normalize_birads("6 nachgewiesene Malignität", norm, "keep") == "6"

    def test_acr_simple(self):
        norm = self.build_normalizer({"1": ["1", "minimal"]})
        val, err = self.normalize_acr("1", norm)
        assert val == "1"
        assert err is False

    def test_acr_textual(self):
        norm = self.build_normalizer({"3": ["3 moderat (acr iii)", "moderat"]})
        val, err = self.normalize_acr("3 Moderat (ACR III)", norm)
        assert val == "3"
        assert err is False

    def test_acr_range_min(self):
        norm = self.build_normalizer({})
        val, err = self.normalize_acr("1 bis 2", norm, acr_range="min")
        assert val == "1"
        assert err is False

    def test_acr_range_max(self):
        norm = self.build_normalizer({})
        val, err = self.normalize_acr("1 bis 2", norm, acr_range="max")
        assert val == "2"
        assert err is False

    def test_acr_range_error(self):
        norm = self.build_normalizer({})
        val, err = self.normalize_acr("2 bis 3", norm, acr_range="error")
        assert val is None
        assert err is True

    def test_acr_none(self):
        norm = self.build_normalizer({})
        val, err = self.normalize_acr(None, norm)
        assert val is None
        assert err is False

    def test_acr_question_mark(self):
        norm = self.build_normalizer({})
        val, err = self.normalize_acr("?", norm)
        assert val is None


# ===========================================================================
# BIRADS-Aggregation (Max ueber Laesionen)
# ===========================================================================

class TestBiradsAggregation:
    """Tests fuer _max_birads aus dem Mamma-Loader."""

    def setup_method(self):
        import pandas as pd
        from loaders.mamma_extraction import _max_birads
        self.max_birads = _max_birads
        self.pd = pd

    def test_max_simple(self):
        s = self.pd.Series(["3", "4", "2"])
        assert self.max_birads(s) == "4"

    def test_max_single(self):
        s = self.pd.Series(["5"])
        assert self.max_birads(s) == "5"

    def test_max_with_missing(self):
        s = self.pd.Series(["2", None, "4", "?"])
        assert self.max_birads(s) == "4"

    def test_max_all_empty(self):
        s = self.pd.Series([None, "?", ""])
        assert self.max_birads(s) is None

    def test_max_empty_series(self):
        s = self.pd.Series([], dtype=str)
        assert self.max_birads(s) is None


# ===========================================================================
# Multiset-Matching (Laesionen)
# ===========================================================================

class TestMultisetPRF:
    """Tests fuer _multiset_prf."""

    def setup_method(self):
        from evaluate import _multiset_prf
        self.multiset_prf = _multiset_prf

    def test_exact_match(self):
        tp, fp, fn = self.multiset_prf(["a", "b", "c"], ["a", "b", "c"])
        assert (tp, fp, fn) == (3, 0, 0)

    def test_partial_match(self):
        tp, fp, fn = self.multiset_prf(["a", "b", "c"], ["a", "b"])
        assert (tp, fp, fn) == (2, 0, 1)

    def test_fp_and_fn(self):
        tp, fp, fn = self.multiset_prf(["a", "b"], ["a", "c"])
        assert (tp, fp, fn) == (1, 1, 1)

    def test_multiset_duplicates(self):
        # GT: 2x a, Model: 1x a → TP=1, FN=1
        tp, fp, fn = self.multiset_prf(["a", "a"], ["a"])
        assert (tp, fp, fn) == (1, 0, 1)

    def test_both_empty(self):
        tp, fp, fn = self.multiset_prf([], [])
        assert (tp, fp, fn) == (0, 0, 0)

    def test_only_fp(self):
        tp, fp, fn = self.multiset_prf([], ["a", "b"])
        assert (tp, fp, fn) == (0, 2, 0)

    def test_only_fn(self):
        tp, fp, fn = self.multiset_prf(["a", "b"], [])
        assert (tp, fp, fn) == (0, 0, 2)


# ===========================================================================
# JSON-Reparatur fuer Arm-Templates
# ===========================================================================

class TestJsonRepair:
    """Tests fuer _repair_json."""

    def setup_method(self):
        from loaders.arm_extraction import _repair_json
        self.repair_json = _repair_json

    def test_repair_missing_comma_before_ossicles(self):
        broken = '{\n  "Fracture": {"finding": false, "citation": ""}\n  "Ossicles": {"finding": false, "citation": ""}\n}'
        repaired = self.repair_json(broken)
        parsed = json.loads(repaired)
        assert "Fracture" in parsed
        assert "Ossicles" in parsed

    def test_valid_json_unchanged(self):
        valid = '{\n  "A": {"finding": true, "citation": "text"},\n  "B": {"finding": false, "citation": ""}\n}'
        repaired = self.repair_json(valid)
        # Valides JSON sollte weiterhin parsbar sein
        parsed = json.loads(repaired)
        assert parsed["A"]["finding"] is True

    def test_repair_preserves_values(self):
        broken = '{\n  "Label1": {"finding": true, "citation": "some text"}\n  "Ossicles": {"finding": false, "citation": ""}\n}'
        repaired = self.repair_json(broken)
        parsed = json.loads(repaired)
        assert parsed["Label1"]["finding"] is True
        assert parsed["Ossicles"]["finding"] is False


# ===========================================================================
# ID-Extraktion aus Bildpfaden (Arm)
# ===========================================================================

class TestArmIdExtraction:
    """Tests fuer _extract_id."""

    def setup_method(self):
        from loaders.arm_extraction import _extract_id
        self.extract_id = _extract_id

    def test_clavicle_filename_mode(self):
        path = "clavicle/ConvertedPNGs/578.png"
        assert self.extract_id(path, "filename") == "578"

    def test_clavicle_id_zero(self):
        path = "clavicle/ConvertedPNGs/0.png"
        assert self.extract_id(path, "filename") == "0"

    def test_elbow_parent_dir_mode(self):
        path = "/data/project/elbow/ConvertedPNGs_AP_Lat/1234/ap.png"
        assert self.extract_id(path, "parent_dir") == "1234"

    def test_thumb_parent_dir_mode(self):
        path = "/data/project/thumb/Images_AP_Lat/567/ap.png"
        assert self.extract_id(path, "parent_dir") == "567"

    def test_filename_without_extension(self):
        # Robustheit: Pfad ohne Extension
        path = "ConvertedPNGs/123"
        assert self.extract_id(path, "filename") == "123"


# ===========================================================================
# Mamma Task JSON-Parsing
# ===========================================================================

class TestMammaJsonParsing:
    """Tests fuer _parse_response im Mamma-Task."""

    def setup_method(self):
        from tasks.mamma_extraction import _parse_response, _extract_side
        self.parse_response = _parse_response
        self.extract_side   = _extract_side

    def test_valid_json(self):
        raw = '{"menopause": "post", "links": {"birads": 3, "acr": 2, "lesionen": ["Fibroadenom"]}, "rechts": {"birads": 4, "acr": 1, "lesionen": []}}'
        parsed, err = self.parse_response(raw)
        assert err is False
        assert parsed["menopause"] == "post"
        assert parsed["links"]["birads"] == 3

    def test_markdown_codeblock(self):
        raw = '```json\n{"menopause": "prä", "links": null, "rechts": null}\n```'
        parsed, err = self.parse_response(raw)
        assert err is False
        assert parsed["menopause"] == "prä"

    def test_error_string(self):
        _, err = self.parse_response("Error: timeout")
        assert err is True

    def test_empty_string(self):
        _, err = self.parse_response("")
        assert err is True

    def test_invalid_json(self):
        _, err = self.parse_response("{invalid json}")
        assert err is True

    def test_extract_side_birads_integer(self):
        parsed = {"links": {"birads": 4, "acr": 2, "lesionen": ["DCIS"]}}
        b, a, l = self.extract_side(parsed, "links")
        assert b == "4"
        assert a == "2"
        assert l == ["DCIS"]

    def test_extract_side_null(self):
        parsed = {"links": None}
        b, a, l = self.extract_side(parsed, "links")
        assert b is None
        assert a is None
        assert l == []

    def test_extract_side_missing_key(self):
        parsed = {}
        b, a, l = self.extract_side(parsed, "rechts")
        assert b is None


# ===========================================================================
# Arm Task JSON-Parsing
# ===========================================================================

class TestArmJsonParsing:
    """Tests fuer _parse_response im Arm-Task."""

    def setup_method(self):
        from tasks.arm_extraction import _parse_response
        self.parse_response = _parse_response

    _LABELS = ["Fracture", "Displacement", "Ossicles"]

    def test_valid_json(self):
        raw = json.dumps({
            "Fracture":    {"finding": True,  "citation": "fracture visible"},
            "Displacement": {"finding": False, "citation": ""},
            "Ossicles":    {"finding": False, "citation": ""},
        })
        parsed, err = self.parse_response(raw, self._LABELS)
        assert err is False
        assert parsed["Fracture"]["finding"] is True
        assert parsed["Displacement"]["finding"] is False

    def test_missing_label_defaults_to_false(self):
        raw = json.dumps({"Fracture": {"finding": True, "citation": "x"},
                          "Displacement": {"finding": False, "citation": ""}})
        parsed, err = self.parse_response(raw, self._LABELS)
        assert err is False
        # Fehlende Labels werden als False ergaenzt
        assert parsed.get("Ossicles", {}).get("finding") is False

    def test_too_few_labels_is_parse_error(self):
        raw = json.dumps({"Fracture": {"finding": True, "citation": "x"}})
        _, err = self.parse_response(raw, self._LABELS)
        assert err is True

    def test_markdown_stripped(self):
        raw = '```\n{"Fracture": {"finding": true, "citation": "test"}}\n```'
        parsed, err = self.parse_response(raw, ["Fracture"])
        assert err is False
        assert parsed["Fracture"]["finding"] is True

    def test_error_response(self):
        _, err = self.parse_response("Error: connection refused", self._LABELS)
        assert err is True


# ===========================================================================
# Bootstrap-CI
# ===========================================================================

class TestBootstrapCI:
    """Tests fuer _bootstrap_ci."""

    def setup_method(self):
        from evaluate import _bootstrap_ci
        self.bootstrap_ci = _bootstrap_ci

    def test_all_ones(self):
        point, lo, hi = self.bootstrap_ci([1.0] * 100)
        assert abs(point - 1.0) < 1e-9
        assert abs(lo - 1.0) < 1e-9
        assert abs(hi - 1.0) < 1e-9

    def test_all_zeros(self):
        point, lo, hi = self.bootstrap_ci([0.0] * 100)
        assert abs(point - 0.0) < 1e-9

    def test_ci_ordering(self):
        values = [float(i % 2) for i in range(200)]
        point, lo, hi = self.bootstrap_ci(values)
        assert lo <= point <= hi

    def test_single_value(self):
        point, lo, hi = self.bootstrap_ci([0.7])
        assert abs(point - 0.7) < 1e-9

    def test_empty(self):
        point, lo, hi = self.bootstrap_ci([])
        assert point == 0.0
