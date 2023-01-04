from typing import Dict, Union
import tensorflow as tf
from connectionist.layers import PMSPLayer


class PMSP(tf.keras.Model):
    """PMSP sim 3 model.

    See Plaut, McClelland, Seidenberg and Patterson (1996), simulation 3 for details.

    When calling the model with `return_internals`=`True`, the model will return a dictionary with all the internal dynamics. Including:

    - `h`: the hidden layer
    - `p`: the phonology output layer
    - `c`: the cleanup layer
    - `oh`: input from o to h, i.e., $o @ w_oh$
    - `ph`: input from p to h, i.e., $p @ w_ph$
    - `hp`: input from h to p, i.e., $h @ w_hp$
    - `pp`: input from p to p, i.e., $p @ w_pp$
    - `cp`: input from c to p, i.e., $c @ w_cp$

    Otherwise, the model will return the phonology output layer, i.e., $p$. This is the default behavior.
    """

    def __init__(self, tau: float, h_units: int, p_units: int, c_units: int) -> None:
        super().__init__()
        self.pmsp = PMSPLayer(
            tau=tau, h_units=h_units, p_units=p_units, c_units=c_units
        )

    def call(
        self, inputs: tf.Tensor, return_internals: bool = False
    ) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:
        return self.pmsp(inputs, return_internals=return_internals)
