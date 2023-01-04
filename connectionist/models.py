from typing import Dict, Union
import tensorflow as tf
from connectionist.layers import PMSPLayer


class PMSP(tf.keras.Model):
    def __init__(self, tau: float, h_units: int, p_units: int, c_units: int) -> None:
        super().__init__()
        self.pmsp = PMSPLayer(
            tau=tau, h_units=h_units, p_units=p_units, c_units=c_units
        )

    def call(
        self, inputs: tf.Tensor, return_internals: bool = False
    ) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:
        return self.pmsp(inputs, return_internals=return_internals)
