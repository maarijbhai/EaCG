"""ecgproto — Python prototype for the paper-ECG digitisation thesis.

The pipeline this package implements is deliberately *classical*: grid
detection, binarisation and column-wise tracing, with no learned
components. It exists as the control condition against which any later
ML stage must justify itself, so it should not be deleted or quietly
replaced once neural components appear.

Module map
----------
ptbxl     loading real 12-lead records and their multi-label diagnoses
synth     synthetic test signals and single-strip printout rendering
layout    clinical 3x4 + rhythm strip printout rendering
deskew    recover and remove in-plane rotation (pipeline stage 2)
digitise  image -> waveform, the classical pipeline
evaluate  SNR / shifted SNR / RMSE / PCC reconstruction metrics
"""

__all__ = ["ptbxl", "synth", "layout", "deskew", "digitise", "evaluate"]
__version__ = "0.1.0"
