from typing import Dict, Union, Tuple, List
import tensorflow as tf
from connectionist.layers import PMSPLayer
from connectionist.damage.shrink_layer import SurgeryPlan, Surgeon, make_recipient


class PMSP(tf.keras.Model):
    """PMSP sim 3 model. A recurrent neural network with hidden layer, phonology layer and cleanup layer. It takes orthography as input and outputs phonology.

    See Plaut, McClelland, Seidenberg and Patterson (1996), simulation 3 for details.

    Args:
        tau: the time constant (smaller time constant = slower temporal dynamics, tau in [0, 1])
        h_units: the number of hidden units
        p_units: the number of phonology units
        c_units: the number of cleanup units
        connections: the connections between layers, default to: ['oh', 'ph', 'hp', 'pp', 'cp', 'pc']

    Call args:

        inputs: the input tensor, shape: (batch_size, time_steps, o_units)
        return_internals: whether to return the internal dynamics, default to: False.

            If `return_internals`=`True`, the model will return a dictionary with all the internal dynamics. Including:
                - activations in all layers (e.g., 'hidden')
                - raw inputs in each connection (e.g., 'oh' = $o @ w_oh$)

            If `return_internals`=`False` the model will return the phonology activation, i.e., $p$.

    Example:

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
    ```

    """

    def __init__(
        self,
        tau: float,
        h_units: int,
        p_units: int,
        c_units: int,
        connections: List[str] = None,
    ) -> None:
        super().__init__()

        self.pmsp = PMSPLayer(
            tau=tau,
            h_units=h_units,
            p_units=p_units,
            c_units=c_units,
            connections=connections,
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
            connections=self.pmsp.connections,
        )

    @property
    def weights_abbreviations(self) -> Dict[str, str]:
        """Weights name abbrevations. (e.g., w_oh, bias_hidden)."""

        weights = {f"w_{w}": f"{w}:0" for w in self.pmsp.connections}

        biases = {
            f"bias_{layer}": f"{layer}/bias:0" for layer in self.pmsp.all_layers_names
        }
        return {**weights, **biases}

    def _find_conn_locs(self, layer: str) -> Dict[str, int]:
        """Find the connection locations of the target layer."""

        def find(l: List[str], prefix: str):
            """Locate connection by layer prefix."""
            return [i for i, x in enumerate(l) if x == prefix]

        prefix = layer[0]
        conn_weights = {}
        for connection in self.pmsp.connections:
            if prefix in connection:
                conn_weights[f"w_{connection}"] = find(connection, prefix)

        conn_bias = {f"bias_{layer}": [0]}
        return {**conn_weights, **conn_bias}

    @property
    def connection_locs(self) -> Dict[str, Dict[str, int]]:
        """A map that shows which axis of a weight is connecting to a layer."""

        return {
            layer: self._find_conn_locs(layer) for layer in self.pmsp.all_layers_names
        }

    def _validate_layer(self, layer: str) -> None:
        """Validate the target layer."""

        layers = list(self.connection_locs.keys())
        if layer not in self.layers:
            raise ValueError(
                f"Unknown target layer: {layer}, please choose from {layers}"
            )

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

        Example:

        ```python
        from connectionist.data import ToyOP
        from connectionist.models import PMSP

        model = PMSP(tau=0.2, h_units=10, p_units=9, c_units=5)

        # Build or call or fit a model to instantiate the weights
        y = model(data.x_train)

        new_model = model.shrink_layer('hidden', rate=0.5)
        ```

        """

        plan = SurgeryPlan(
            layer=layer, original_units=self.get_units(layer), shrink_rate=rate
        )
        surgeon = Surgeon(surgery_plan=plan)

        new_model = make_recipient(model=self, surgery_plan=plan, make_model_fn=PMSP)
        new_model.build(input_shape=self._saved_model_inputs_spec.shape)

        surgeon.transplant(donor=self, recipient=new_model)
        return new_model

    def cut_connection(self, connection: str) -> None:
        """Cut a connection between two layers."""

        pass
