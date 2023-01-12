import numpy as np
import tensorflow as tf


class ToyOP:
    """Toy O-to-P dataset containing only 20 examples.

    - x_train: word representations, fixed across time.
    - y_train: letter sequence representations, changing across time.
    """

    letters = "crsaoutbn"
    words = [
        "cat",
        "cab",
        "can",
        "cot",
        "cob",
        "con",
        "cut",
        "cub",
        "rat",
        "ran",
        "rot",
        "rob",
        "rut",
        "rub",
        "run",
        "sat",
        "sot",
        "sob",
        "son",
        "sun",
    ]

    def __init__(self, max_ticks: int = 30):
        self.max_ticks = max_ticks
        self.x_train, self.y_train = self.get_data()

        if max_ticks % 3 != 0:
            raise ValueError("max_ticks must be divisible by 3")

    @property
    def n(self) -> int:
        return len(self.words)

    def get_data(self):
        word_repr = np.zeros((self.n, self.n))
        spelling_repr = np.zeros((self.n, len(self.words[0]), len(self.letters)))

        # Encode
        for i, word in enumerate(self.words):
            word_repr[i][i] = 1.0
            for j, letter in enumerate(word):
                spelling_repr[i][j][self.letters.index(letter)] = 1

        # Repeat over time axis
        x_train = np.stack([word_repr for _ in range(self.max_ticks)], axis=1)
        y_train = np.repeat(spelling_repr, self.max_ticks / 3, axis=1)

        return (
            tf.convert_to_tensor(x_train, dtype=tf.int8),
            tf.convert_to_tensor(y_train, dtype=tf.int8),
        )

    def __repr__(self) -> str:
        return f"Time invariant orthographic word representation: {self.x_train.shape=}.\nTime varying letter/phoneme sequence representation{self.y_train.shape=}"
