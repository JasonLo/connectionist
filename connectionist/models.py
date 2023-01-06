from typing import Dict, Union, Tuple, List
import tensorflow as tf
from connectionist.layers import PMSPLayer
from connectionist.damage.shrink_layer import SurgeryPlan, Surgeon, make_recipient


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

    def get_config(self) -> Dict[str, Union[float, int]]:
        return dict(
            tau=self.pmsp.tau,
            h_units=self.pmsp.h_units,
            p_units=self.pmsp.p_units,
            c_units=self.pmsp.c_units,
        )

    @property
    def abbreviations(self) -> Dict[str, str]:
        """Weights name abbrevations."""

        return {
            "w_oh": "pmsp_cell/o2h/kernel",
            "w_ph": "pmsp_cell/p2h/kernel",
            "w_hp": "pmsp_cell/h2p/kernel",
            "w_pp": "pmsp_cell/p2p/kernel",
            "w_cp": "pmsp_cell/c2p/kernel",
            "w_pc": "pmsp_cell/p2c/kernel",
            "bias_h": "pmsp_cell/ta_h/bias",
            "bias_p": "pmsp_cell/ta_p/bias",
            "bias_c": "pmsp_cell/p2c/bias",
        }

    @property
    def connections(self) -> Dict[str, Dict[str, Union[int, List[int]]]]:
        """A map that shows which axis of a weight is connecting to a layer."""

        return {
            "hidden": {"w_oh": 1, "w_ph": 1, "w_hp": 0, "bias_h": 0},
            "cleanup": {"w_cp": 0, "w_pc": 1, "bias_c": 0},
            "phonology": {
                "w_hp": 1,
                "w_pp": [0, 1],
                "w_cp": 1,
                "w_pc": 0,
                "w_ph": 0,
                "bias_p": 0,
            },
        }

    def _validate_layer(self, layer: str) -> None:
        """Validate the target layer."""

        layers = list(self.connections.keys())
        if layer not in self.layers:
            raise ValueError(
                f"Unknown target layer: {layer}, please choose from {layers}"
            )

    def locate_connections(
        self, layer: str
    ) -> Tuple[List[str], List[Union[int, List[int]]]]:
        """Get the lesion locations based on the target layer.

        Returns:
        - name abbrevations
        - axis to correspond to the target layer
        """

        connections = self.connections[layer]

        # convert the name abbrevations to the full names
        names = [self.abbreviations[name] for name in connections.keys()]
        return names, list(connections.values())

    def get_units(self, layer: str) -> int:
        """Get the number of units in the target layer.

        Args:
            layer: the target layer, choose from ['hidden', 'phonology', 'cleanup']
        """

        mapping = {
            "hidden": self.pmsp.h_units,
            "phonology": self.pmsp.p_units,
            "cleanup": self.pmsp.c_units,
        }
        return mapping[layer]

    def shrink_layer(self, layer: str, rate: float) -> None:
        """Shrink the weights of the target layer.

        Args:
            layer: the target layer, choose from ['hidden', 'phonology', 'cleanup']
            rate: the shrink rate
        """

        plan = SurgeryPlan(
            layer=layer, original_units=self.get_units(layer), shrink_rate=rate
        )
        surgeon = Surgeon(surgery_plan=plan)

        new_model = make_recipient(model=self, surgery_plan=plan, make_model_fn=PMSP)
        new_model.build(input_shape=self._saved_model_inputs_spec.shape)

        surgeon.transplant(donor=self, recipient=new_model)
        return new_model

    def cut_connection(self, weight: str) -> None:
        """Cut a connection between two layers."""

        pass
