#!/usr/bin/env python3
"""Bundle the printed kit sheets' ground truth into the Android app.

So that the handset can score a photograph of a printed `make_kit_target.py`
sheet against the PTB-XL record it was printed from -- the same comparison
`score_kit_photo.py` makes on the desktop, made by the phone itself. A
fidelity figure carried across from the desktop could not detect a porting
bug; the device has to hold the truth to check its own answer.

Writes, under android/app/src/main/assets/kit/:

    targets.txt          one line per sheet, `key=value` pairs: name,
                         ecg_id, strat_fold, the printed sheet's size in mm
                         (exact, from its pixel size and dpi), the truth's
                         sampling rate and length, and the truth file name.
    <name>_II.csv        the lead II rhythm strip, one mV value per line,
                         PTB-XL at 500 Hz -- all 10 s, which is what the
                         rhythm strip prints.
    <name>_12lead.csv    all 12 leads, 10 s at 500 Hz, a header line of
                         lead names (the printout's spelling: aVR, not AVR)
                         then one comma-separated row per sample. Each 3x4
                         cell is scored against its own 2.5 s window of its
                         lead (the full-sheet path, SheetPipeline.kt).

Flat text for the same reason `TruthWaveform` reads flat text: a parser for
.npy or JSON on the device is a porting hazard in exchange for kilobytes.

    python make_kit_assets.py            # every sheet in out/kit_targets
"""

from __future__ import annotations

from pathlib import Path

from ecgproto import ptbxl
from score_kit_photo import TARGETS, Target

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "android" / "app" / "src" / "main" / "assets" / "kit"


def main() -> int:
    names = sorted(d.name for d in TARGETS.glob("ptbxl*_kit_*") if d.is_dir())
    if not names:
        print(f"no kit targets in {TARGETS}; run make_kit_target.py first")
        return 2
    ASSETS.mkdir(parents=True, exist_ok=True)
    lines = ["# written by python/make_kit_assets.py; one printed sheet per line"]
    for name in names:
        t = Target(name)
        csv = f"{name}_II.csv"
        (ASSETS / csv).write_text(
            "\n".join(repr(float(v)) for v in t.lead_signal) + "\n",
            encoding="utf-8")
        csv12 = f"{name}_12lead.csv"
        names12 = ["I", "II", "III", "aVR", "aVL", "aVF",
                   "V1", "V2", "V3", "V4", "V5", "V6"]
        assert [n.upper() for n in names12] == list(ptbxl.LEAD_NAMES)
        rows12 = [",".join(names12)] + [
            ",".join(repr(float(v)) for v in t.truth[i]) for i in range(len(t.truth))]
        (ASSETS / csv12).write_text("\n".join(rows12) + "\n", encoding="utf-8")
        lines.append(" ".join([
            f"name={name}",
            f"ecg_id={t.ecg_id}",
            f"strat_fold={t.manifest['source']['strat_fold']}",
            f"width_mm={t.width_mm!r}",
            f"height_mm={t.height_mm!r}",
            f"fs={t.fs!r}",
            f"n={len(t.lead_signal)}",
            f"truth={csv}",
            f"truth12={csv12}",
        ]))
        print(f"  {name}: {len(t.lead_signal)} samples, "
              f"{t.width_mm:.3f} x {t.height_mm:.3f} mm")
    (ASSETS / "targets.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(names)} sheet(s) -> {ASSETS.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
