# Connectionist

[![Pypi version](https://img.shields.io/pypi/v/connectionist.svg?style=flat&color=brightgreen)](https://pypi.org/project/connectionist/)
[![Documentation Status](https://readthedocs.org/projects/connectionist/badge/?version=latest)](https://connectionist.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/github/license/jasonlo/connectionist.svg?color=brightgreen)](https://github.com/JasonLo/connectionist/blob/main/LICENSE)

Tools for classical connectionist models of reading with TensorFlow.

## Requirements

- Python >=3.8
- TensorFlow >=2.9

## Installation

```bash
pip install connectionist
```

## Quick start

End-to-end toy example with Plaut, McClelland, Seidenberg and Patterson (1996), simulation 3 model:

```python
import tensorflow as tf
from connectionist.data import ToyOP
from connectionist.models import PMSP

data = ToyOP()
model = PMSP(tau=0.2, h_units=10, p_units=9, c_units=5)
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(),
)
model.fit(data.x_train, data.y_train, epochs=3, batch_size=20)
model(data.x_train)
```

## Modules

- connectionist.data: Includes datasets for connectionist models of reading. Currently only have `ToyOP`, but will add more in the future.
- connectionist.layers: Includes custom layers for connectionist models of reading in `tf.keras.layers.Layer` format.
- connectionist.models: Includes ready-to-use connectionist models in `tf.keras.Model` format.
- connectionist.losses: Includes custom losses functions for connectionist models of reading.
- connectionist.surgery: Includes helper functions for "brain damage" experiments.

## Documentation

[![Github](https://img.shields.io/badge/docs-Github.io-4051b5)](https://jasonlo.github.io/connectionist/)
[![Read the Docs](https://img.shields.io/badge/docs-Read%20the%20Docs-4051b5)](https://connectionist.readthedocs.io/en/latest/)
