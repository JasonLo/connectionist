from typing import Dict, Union, Tuple, List
import tensorflow as tf
from connectionist.layers import PMSPLayer
from connectionist.surgery import SurgeryPlan, Surgeon, make_recipient


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
        config = super().get_config()
        config.update(
            tau=self.pmsp.tau,
            h_units=self.pmsp.h_units,
            p_units=self.pmsp.p_units,
            c_units=self.pmsp.c_units,
        )
        return config

    @property
    def abbreviations(self) -> Dict[str, str]:
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

    def layer2weights(self, layer: str) -> Tuple[List[str], List[int]]:
        """Get the lesion locations based on the target layer.

        Returns:
        - name abbrevations
        - axis to correspond to the target layer
        """

        assert layer in [
            "hidden",
            "phonology",
            "cleanup",
        ], f"Unknown target layer: {layer}, please choose from ['hidden', 'phonology', 'cleanup']"

        if layer == "hidden":
            short_names = ["w_oh", "w_ph", "w_hp", "bias_h"]
            axes = [1, 1, 0, 0]

        if layer == "phonology":
            short_names = [
                "w_hp",
                "w_pp",
                "w_cp",
                "w_pc",
                "w_ph",
                "bias_p",
                "w_pp",
            ]  # w_pp need both axis 0 and 1
            axes = [1, 1, 1, 0, 0, 0, 0]

        if layer == "cleanup":
            short_names = ["w_cp", "w_pc", "bias_c"]
            axes = [1, 0, 0]

        names = [self.abbreviations[short_name] for short_name in short_names]
        return names, axes

    def get_units(self, layer: str) -> int:
        """Get the units of the target layer.

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
        new_model.build(input_shape=self._build_input_shape)

        surgeon.transplant(donor=self, recipient=new_model)
        return new_model
