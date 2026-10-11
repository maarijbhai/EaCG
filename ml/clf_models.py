"""Classifier architectures for the diagnosis experiment (ml/train_classifier.py).

resnet   A small 1-D ResNet: a strided stem and four stages of two residual
         blocks (kernel 7), global average + max pooling, a dense head.
         ~0.5 M parameters -- small enough to run on the target phone later
         beside the tracer (0.39 M), large enough for PTB-XL, where 1-D
         ResNets are among the strongest published models.
mishra   A re-implementation of Mishra et al. (2021, J. Med. Biol. Eng.
         41:422) Fig. 3b as described: one convolution (16 filters, size 3)
         + ReLU, fully connected layers (384, then a hidden layer), and a
         softmax over 5 classes, on a 400-point single-lead input. Their
         paper gives no more detail (padding, hidden width, dropout); the
         choices here are stated in the code. A baseline, not their model.

Both end in a Dense layer with no activation; the training script applies
sigmoid (multi-label) or softmax (single-label) in the loss.
"""

from __future__ import annotations


def resnet1d(input_shape, n_out: int, width=(32, 64, 96, 128), kernel: int = 7, dropout: float = 0.3):
    import keras
    from keras import layers as L

    def block(x, ch, stride):
        y = L.Conv1D(ch, kernel, strides=stride, padding="same", use_bias=False)(x)
        y = L.BatchNormalization()(y)
        y = L.ReLU()(y)
        y = L.Conv1D(ch, kernel, padding="same", use_bias=False)(y)
        y = L.BatchNormalization()(y)
        if stride != 1 or x.shape[-1] != ch:
            x = L.Conv1D(ch, 1, strides=stride, use_bias=False)(x)
            x = L.BatchNormalization()(x)
        return L.ReLU()(L.Add()([x, y]))

    inp = keras.Input(input_shape, name="ecg")
    x = L.Conv1D(width[0], 15, strides=2, padding="same", use_bias=False)(inp)
    x = L.BatchNormalization()(x)
    x = L.ReLU()(x)
    for i, ch in enumerate(width):
        x = block(x, ch, 1 if i == 0 else 2)
        x = block(x, ch, 1)
    x = L.Concatenate()([L.GlobalAveragePooling1D()(x), L.GlobalMaxPooling1D()(x)])
    x = L.Dropout(dropout)(x)
    out = L.Dense(n_out, dtype="float32", name="logits")(x)
    return keras.Model(inp, out, name="resnet1d")


def mishra(input_shape, n_out: int, hidden: int = 128, dropout: float = 0.3):
    """Mishra et al. Fig. 3b: conv(16, 3) + ReLU -> FC 384 -> FC hidden -> 5.

    Stated choices (not in the paper): 'same' padding (they mention it),
    hidden width 128, dropout 0.3 before the output.
    """
    import keras
    from keras import layers as L
    inp = keras.Input(input_shape, name="ecg")
    x = L.Conv1D(16, 3, padding="same", activation="relu")(inp)
    x = L.Flatten()(x)
    x = L.Dense(384, activation="relu")(x)
    x = L.Dense(hidden, activation="relu")(x)
    x = L.Dropout(dropout)(x)
    out = L.Dense(n_out, dtype="float32", name="logits")(x)
    return keras.Model(inp, out, name="mishra")


ARCHS = {"resnet": resnet1d, "mishra": mishra}
