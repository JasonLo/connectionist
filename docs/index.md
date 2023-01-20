---
hide:
  - navigation
---

# Welcome to Connectionist's documentation

**Connectionist** contains some tools for classical [connectionist models of reading](https://www.cnbc.cmu.edu/~plaut/papers/pdf/Plaut05chap.connModelsReading.pdf) in TensorFlow. This project is a course companion python library for [Contemporary neural networks for cognition and cognitive neuroscience](https://drive.google.com/drive/folders/1ZNmK-W8bk3iIH6M5cYzhO_XGhCrxFXzL).

## Features

- Ready-to-use models of reading in TensorFlow
- Various "brain" (model) damaging APIs
- Basic building blocks (layers) for connectionist models

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
