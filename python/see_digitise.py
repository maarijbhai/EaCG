import sys, json, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2, numpy as np, wfdb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ecgproto import digitise as dig, evaluate

STYLE, STEM = "poc_clean", "00009_hr"        # <-- change these
base = Path(__file__).resolve().parents[1] / "data" / "images" / STYLE
img  = cv2.imread(str(base / f"{STEM}-0.png"), cv2.IMREAD_COLOR)
meta = json.loads((base / f"{STEM}-0.json").read_text())

# 0. Make sure the output directory exists
out_dir = Path("out")
out_dir.mkdir(exist_ok=True)

# 1. Pick the rhythm strip — the ONLY lead that is one continuous 10 s trace.
#    The other 12 are 2.5 s cells; digitise() would trace across their borders.
strip = [L for L in meta["leads"]
         if L["end_sample"] - L["start_sample"] == 5000][0]
pts   = np.array(strip["plotted_pixels"])          # [y, x] — not [x, y]
ys    = [strip["lead_bounding_box"][k][0] for k in "0123"]

pad = int(round(2 * meta["x_grid"]))               # ~10 mm of paper above/below
y0, y1 = max(0, int(min(ys)) - pad), min(img.shape[0], int(max(ys)) + pad)
x0, x1 = int(pts[:, 1].min()), int(pts[:, 1].max())   # starts after the DC pulse
crop = img[y0:y1, x0:x1]

# 2. Grid pitch. On these images the 1 mm rules are ~7x fainter than the 5 mm
#    ones, so a threshold above ~0.10 filters them out and the pitch collapses
#    onto the majors: 39.26 px/mm against a true 7.874, a silent 5x error.
#    The default is 0.05 as of 2026-09-17 and now handles this; 0.20 is pinned
#    below to keep showing what it used to do.
truth_pxmm = meta["x_grid"] / 5.0
g = dig.estimate_grid_pitch(crop)
old_default = dig.estimate_grid_pitch(crop, peak_fraction=0.20).px_per_mm
print(f"true px/mm {truth_pxmm:.4f} | recovered {g.px_per_mm:.4f} "
      f"({(g.px_per_mm - truth_pxmm) / truth_pxmm * 100:+.2f}%) | "
      f"the old 0.20 default would give {old_default:.2f}")

# 3. Digitise.
res = dig.digitise(crop, fs_out=meta["sampling_frequency"], grid=g)

# 4. Ground truth: the companion .dat holds exactly what was printed, in mV.
rec = wfdb.rdrecord(str(base / STEM))
sig = rec.p_signal
# Prefer lead II by name if we can find it, else fall back to column 1 (or 0).
lead_names = [n.strip().upper() for n in (rec.sig_name or [])]
if "II" in lead_names:
    truth = sig[:, lead_names.index("II")]
elif sig.shape[1] > 1:
    truth = sig[:, 1]        # assume [I, II, III, ...]
else:
    truth = sig[:, 0]        # single-column rhythm strip

fs = meta["sampling_frequency"]
m = evaluate.evaluate_all(truth, res.signal_mv, fs=fs)
print(f"SNR {m.snr_db:.2f} dB | RMSE {m.rmse_mv:.4f} mV | PCC {m.pcc:.4f} | "
      f"gain {m.gain_ratio:.3f} | coverage {res.coverage:.1%}")
print(m.diagnose())

# 5. Look at it.
y_px = dig.trace_positions(dig.binarise_trace(crop))
fig, ax = plt.subplots(3, 1, figsize=(16, 10))
ax[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
ax[0].set_title("cropped strip")
ax[1].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
ax[1].plot(np.arange(len(y_px)), y_px, lw=0.8, color="lime")
ax[1].set_title("what the tracer followed")

n = min(len(truth), len(res.signal_mv))
t = np.arange(n) / fs
ax[2].plot(t, truth[:n], lw=0.9, label="ground truth")
ax[2].plot(t, res.signal_mv[:n], lw=0.9, alpha=.8, label="digitised")
ax[2].legend(); ax[2].set_xlabel("s"); ax[2].set_ylabel("mV")
ax[2].set_title(f"PCC {m.pcc:.3f} · gain {m.gain_ratio:.3f} · RMSE {m.rmse_mv:.4f} mV")
for a in ax[:2]:
    a.axis("off")
plt.tight_layout()
plt.savefig(out_dir / "see_digitise.png", dpi=110)
print("wrote out/see_digitise.png")