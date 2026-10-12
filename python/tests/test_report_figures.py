"""make_report_figures: parsing, the example-lead rule, failures counted, pairing."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import make_report_figures as R  # noqa: E402


def frame(rows):
    df = pd.DataFrame(rows)
    for c, v in (("rhythm", False), ("set", "ideal"), ("condition", "ideal"), ("ecg_id", 40),
                 ("pcc_aligned", np.nan)):
        if c not in df:
            df[c] = v
    df["rhythm"] = df["rhythm"].fillna(False).astype(bool)
    df["available"] = df["pcc_asis"].notna()
    return df


class Parsing(unittest.TestCase):

    def test_timings_new_old_and_missing(self):
        self.assertEqual(R.parse_timings("corners=0 rectify=224 cut=33 score*=96"),
                         {"corners": 0, "rectify": 224, "cut": 33, "score*": 96})
        old = R.parse_timings("corners=0 rectify=175 grid=2051 trace=19425 cut+score=7570")
        self.assertEqual(old["cut+score"], 7570)
        self.assertEqual(R.parse_timings(None), {})
        self.assertEqual(R.parse_timings(float("nan")), {})
        self.assertEqual(R.parse_timings("corners=0 broken= x=y"), {"corners": 0})

    def test_stage_kinds(self):
        self.assertEqual(R.stage_kind("trace"), "pipeline")
        self.assertEqual(R.stage_kind("score*"), "harness")
        self.assertEqual(R.stage_kind("score"), "harness")
        self.assertEqual(R.stage_kind("cut+score"), "legacy")

    def test_memory_and_missing(self):
        base, st = R.parse_memory("baseline=59.7/37.2 grid=141.5/77.9 trace=510.6/65.8")
        self.assertEqual(base, (59.7, 37.2))
        self.assertEqual(st["trace"], (510.6, 65.8))
        self.assertEqual(R.parse_memory(""), (None, {}))
        self.assertEqual(R.parse_memory(float("nan")), (None, {}))

    def test_tracer_and_group(self):
        self.assertEqual(R.tracer_kind("model ml/b3_512x1024_fp16.tflite"), "learned")
        self.assertEqual(R.tracer_kind("classical"), "classical")
        self.assertEqual(R.group_of(38), "dev")
        self.assertEqual(R.group_of(4016), "held_out")
        self.assertEqual(R.group_of(12345), "other")


class ExampleSelection(unittest.TestCase):

    def test_picks_the_median_nearest_learned_lead(self):
        df = frame([
            {"export": "a", "photo_id": "p1", "lead": "I", "tracer_kind": "learned", "pcc_asis": 0.99},
            {"export": "a", "photo_id": "p1", "lead": "V1", "tracer_kind": "learned", "pcc_asis": 0.80},
            {"export": "b", "photo_id": "p2", "lead": "II", "tracer_kind": "learned", "pcc_asis": 0.94},
            {"export": "b", "photo_id": "p2", "lead": "aVR", "tracer_kind": "learned", "pcc_asis": 0.97},
            # classical and rhythm rows never chosen, a failure never chosen
            {"export": "c", "photo_id": "p1", "lead": "I", "tracer_kind": "classical", "pcc_asis": 0.955},
            {"export": "a", "photo_id": "p1", "lead": "II", "tracer_kind": "learned", "pcc_asis": 0.955, "rhythm": True},
            {"export": "a", "photo_id": "p1", "lead": "V2", "tracer_kind": "learned", "pcc_asis": np.nan},
        ])
        df.loc[5, "rhythm"] = True
        pick = R.choose_example(df)
        # learned non-rhythm available: 0.99, 0.80, 0.94, 0.97 -> median 0.955; nearest 0.94 / 0.97 tie
        # by distance 0.015 each -> tie broken by export name: 'b' both; then lead order II before aVR.
        self.assertEqual((pick["photo_id"], pick["lead"]), ("p2", "II"))

    def test_no_learned_leads_gives_none(self):
        df = frame([{"export": "c", "photo_id": "p", "lead": "I", "tracer_kind": "classical", "pcc_asis": 0.9}])
        self.assertIsNone(R.choose_example(df))


class Failures(unittest.TestCase):

    def test_empty_pcc_is_a_failure_in_n_and_pct(self):
        sm = R.summary([0.95, 0.92, np.nan, 0.5])
        self.assertEqual(sm["n"], 4)
        self.assertEqual(sm["n_unavailable"], 1)
        self.assertAlmostEqual(sm["pct_ge_09"], 50.0)        # 2 of 4, the failure counts as 'not >= 0.9'
        self.assertAlmostEqual(sm["median"], 0.92)           # median over the available leads

    def test_all_failed(self):
        sm = R.summary([np.nan, np.nan])
        self.assertEqual((sm["n"], sm["n_unavailable"], sm["pct_ge_09"]), (2, 2, 0.0))
        self.assertTrue(np.isnan(sm["median"]))


class Json(unittest.TestCase):

    def test_nan_becomes_null(self):
        import json
        o = R.clean_json({"a": float("nan"), "b": [1.0, np.inf, np.float64(2.5)], "c": np.int64(3)})
        self.assertEqual(json.loads(json.dumps(o, allow_nan=False)), {"a": None, "b": [1.0, None, 2.5], "c": 3})


class Pairing(unittest.TestCase):

    def test_unpaired_learned_export_is_reported_not_crashed_on(self):
        rows = []
        for ld, a, b in (("I", 0.95, 0.80), ("II", 0.97, 0.70), ("III", 0.90, 0.60)):
            rows.append({"export": "L1", "photo_id": "p1", "lead": ld, "tracer_kind": "learned", "pcc_asis": a})
            rows.append({"export": "C1", "photo_id": "p1", "lead": ld, "tracer_kind": "classical", "pcc_asis": b})
        rows.append({"export": "L2", "photo_id": "p2", "lead": "I", "tracer_kind": "learned", "pcc_asis": 0.99})
        pairs, rep = R.paired_tracers(frame(rows))
        self.assertEqual(len(pairs), 3)
        self.assertEqual(rep["learned_photos_without_classical"], ["p2"])
        w = R.wilcoxon(pairs)
        self.assertEqual(w["n_pairs"], 3)
        self.assertIsNotNone(w["p"])

    def test_no_pairs_at_all(self):
        rows = [{"export": "L", "photo_id": "p", "lead": "I", "tracer_kind": "learned", "pcc_asis": 0.9}]
        pairs, rep = R.paired_tracers(frame(rows))
        self.assertEqual(len(pairs), 0)
        self.assertEqual(R.wilcoxon(pairs)["n_pairs"], 0)

    def test_cluster_ci_needs_two_photos(self):
        df = frame([{"export": "a", "photo_id": "p", "lead": "I", "tracer_kind": "learned", "pcc_asis": 0.9}])
        self.assertTrue(all(np.isnan(R.cluster_ci(df, "pcc_asis"))))
        df = frame([{"export": e, "photo_id": e, "lead": "I", "tracer_kind": "learned", "pcc_asis": v}
                    for e, v in (("a", 0.9), ("b", 0.8), ("c", 0.95))])
        lo, hi = R.cluster_ci(df, "pcc_asis")
        self.assertTrue(0.8 <= lo <= hi <= 0.95)


if __name__ == "__main__":
    unittest.main()
