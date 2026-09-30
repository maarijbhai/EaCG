"""Print whether TensorFlow sees a GPU, and run one op on it.
    bash ml/wsl_run.sh python ml/check_gpu.py [--cv2-first]
`--cv2-first` imports OpenCV before TensorFlow, as train_tracer.py must.
"""
import os
import sys

if "--cv2-first" in sys.argv:
    import cv2  # noqa: F401
if "--fork-first" in sys.argv:
    from multiprocessing import get_context
    pool = get_context("fork").Pool(2)
    print("pool:", pool.map(abs, [-1, -2]))
print("LD_LIBRARY_PATH has nvidia:", "nvidia" in os.environ.get("LD_LIBRARY_PATH", ""))

import tensorflow as tf  # noqa: E402

gpus = tf.config.list_physical_devices("GPU")
print("gpus:", gpus)
if gpus:
    with tf.device("/GPU:0"):
        x = tf.random.normal((2048, 2048))
        print("matmul on", (x @ x).device)
