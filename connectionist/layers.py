from typing import Dict, List, Optional, Tuple, Union
from functools import partial
import tensorflow as tf


def _time_averaging(
    x: tf.Tensor, tau: float, states: Optional[tf.Tensor] = None
) -> tf.Tensor:
    r"""Time-averaging mechanism.

    Args:
        x (tf.Tensor): Input tensor.
        tau (float): Time-averaging parameter (How much information should take from the new input). range: [0, 1].
        states (tf.Tensor, optional): Last states (last activation: a_{t-1} or last integrated input: x_{t-1} = \sum ). Defaults to None.
    """

    if states is None:
        return x * tau

    return x * tau + (1 - tau) * states


def reshape_proper(a: tf.TensorArray, perm: List[int] = None) -> tf.TensorArray:
    """Reshape the TensorArray to the standard output shape of [batch_size, sequence_length, feature].

    Args:
        a (tf.TensorArray): TensorArray to be reshaped.
        perm: Permutation of the dimensions of the input TensorArray. Defaults to [1, 0, 2] for typical RNN use case.
    """
    if perm is None:
        perm = [1, 0, 2]
    return tf.transpose(a.stack(), perm)


class TimeAveragedDense(tf.keras.layers.Dense):
    r"""This layer simulates continuous-temporal dynamics with time-averaged input/output.

    Args:
        tau (float): Time-averaging parameter, from 0 to 1.
        average_at (str): Select where to average, 'before_activation' or 'after_activation'.
        kwargs (str, optional): Any argument in `keras.layers.Dense`.

    ### Time-averaged input

    Defines as:

    ```math
    a_t = act(s_t)
    ```
    ```math
    s_t = \tau \cdot (x_t w + b) + (1-\tau) \cdot s_{t-1}
    ```

    - $`a_t`$: activation at time $`t`$
    - $`s_t`$: state at time $`t`$
    - $`\tau`$: time constant, smaller means slower temporal dynamics.
    - $`x_t`$: input at time $`t`$
    - $`w`$: weight matrix (provided by this layer)
    - $`b`$: bias vector (provided by this layer)

    Code example:

    ```python
    layer = TimeAveragedDense(tau=0.1, average_at="before_activation", units=10)
    ```

    ### Time-averaged output

    Defines as:

    ```math
    a_t = \tau \cdot act(x_t w + b) + (1-\tau) \cdot a_{t-1}
    ```

    - $`a_t`$: activation at time $`t`$
    - $`\tau`$: time constant, smaller means slower temporal dynamics.
    - $`x_t`$: input at time $`t`$
    - $`w`$: weight matrix (provided by this layer)
    - $`b`$: bias vector (provided by this layer)

    Code example:

    ```python
    layer = TimeAveragedDense(tau=0.1, average_at="after_activation", units=10)
    ```

    !!! note
        This layer is only useful for single input, for multiple inputs equivalent, see [MultiInputTimeAveraging][connectionist.layers.MultiInputTimeAveraging].

    """

    def __init__(
        self,
        tau: float,
        average_at: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tau = tau
        self.average_at = average_at

        # Turn off activation, will manually call the ._activation in `.call()`.
        self._activation = self.activation
        self.activation = None

        if self.tau < 0 or self.tau > 1:
            raise ValueError(f"tau must be between 0 and 1, but got {self.tau}")

        if self.average_at not in ["before_activation", "after_activation"]:
            raise ValueError(
                f"average_at must be one of ['before_activation', 'after_activation'], but got {self.average_at}"
            )

        self.states = None

    def call(self, inputs: tf.Tensor) -> tf.Tensor:

        # keras.layers.Dense.call() without activation
        outputs = super().call(inputs)
        if self.average_at == "before_activation":
            outputs = _time_averaging(outputs, self.tau, self.states)
            self.states = outputs  # state is integrated input here

        if self._activation is not None:
            outputs = self._activation(outputs)

        if self.average_at == "after_activation":
            outputs = _time_averaging(outputs, self.tau, self.states)
            self.states = outputs  # state is activation here

        return outputs

    def reset_states(self) -> None:
        self.states = None

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "tau": self.tau,
                "average_at": self.average_at,
                "activation": tf.keras.activations.serialize(self._activation),
            }
        )
        return config


class ZeroOutDense(tf.keras.layers.Dense):
    """Dense layer with zero-out (weight masking) mechanism."""

    def __init__(self, zero_out_rate: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.zero_out_rate = zero_out_rate

    def build(self, input_shape: tf.TensorShape) -> None:
        super().build(input_shape)

        # Create a mask to zero-out the weights.
        self.zero_out_mask = tf.Variable(
            tf.ones(self.kernel.shape),
            trainable=False,
            dtype=tf.float32,
            name="zero_out_mask",
        )

        self.zero_out_mask.assign(
            tf.where(
                tf.random.uniform(self.kernel.shape) < self.zero_out_rate,
                tf.zeros(self.kernel.shape, dtype=tf.float32),
                tf.ones(self.kernel.shape, dtype=tf.float32),
            )
        )

        # Also zero-out the initialized values.
        self.kernel.assign(self.kernel * self.zero_out_mask)

    def zero_out_weights(self) -> None:
        """Manually zero-out the weights."""
        self.kernel.assign(self.kernel * self.zero_out_mask)

    def call(self, inputs):

        # Masked kernel is require to makes masked weight's gradient zero during back-propagation.
        # Because: d_kernel = zero_out_mask * d_masked_kernel
        masked_kernel = self.kernel * self.zero_out_mask
        outputs = inputs @ masked_kernel

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"zero_out_rate": self.zero_out_rate})
        return config


class MultiInputTimeAveraging(tf.keras.layers.Layer):
    r"""Time-averaging mechanism for multiple inputs.

    In short, time-averaging mechanism simulates continuous-temporal dynamics in a discrete-time recurrent neural networks.
    See Plaut, McClelland, Seidenberg, and Patterson (1996) equation (15) for more details.

    This layer is designed for multiple inputs, assuming they had ALREADY been multiplied by weights i.e., a list of (x @ w).
    i.e., there is only bias term in this layer (no weights) if use_bias is True.

    Args:
        tau (float): Time-averaging parameter (How much information should take from the new input). range: [0, 1].

        average_at (str): Where to average. Options: 'before_activation', 'after_activation'.

            When average_at is 'before_activation', the time-averaging is applied BEFORE activation. i.e., time-averaging INPUT.:
                outputs = activation(integrated_input);
                integrated input = tau * (sum(inputs) + bias) + (1-tau) * last_integrated_input;
                last_integrated_input is obtained from the last call of this layer, its values stored at `self.states`

            When average_at is 'after_activation', the time-averaging is applied AFTER activation. i.e., time-averaging OUTPUT.:
                outputs = tau * activation(sum(inputs) + bias) + (1-tau) * last_outputs;
                last_outputs is obtained from the last call of this layer, its values stored at `self.states`

        activation (str, optional): Activation function to use. Defaults to None.

        use_bias (bool, optional): Whether the layer uses a bias vector. Defaults to True.

        bias_initializer (optional): Initializer for the bias vector. Defaults to 'zeros'.

        bias_regularizer (optional): Regularizer function applied to the bias vector. Defaults to None.

        bias_constraint (optional): Constraint function applied to the bias vector. Defaults to None.

    """

    def __init__(
        self,
        tau: float,
        average_at: str,
        activation: str = None,
        use_bias: bool = True,
        bias_initializer: str = "zeros",
        bias_regularizer: str = None,
        bias_constraint: str = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tau = tau
        self.average_at = average_at
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        if self.tau < 0 or self.tau > 1:
            raise ValueError(f"tau must be between 0 and 1, but got {self.tau}")

        if self.average_at not in ["before_activation", "after_activation"]:
            raise ValueError(
                f"average_at must be one of ['before_activation', 'after_activation'], but got {self.average_at}"
            )

        self.states = None

    def build(self, input_shape) -> None:

        self.sum = tf.keras.layers.Add()

        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=[
                    input_shape[0][-1],
                ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
            )
        else:
            self.bias = None

        self.built = True

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:

        # Check if all inputs have the same shape
        if not all([x.shape == inputs[0].shape for x in inputs]):
            raise ValueError("All inputs must have the same shape.")

        if len(inputs) == 1:
            outputs = inputs[0]
        else:
            outputs = self.sum(inputs)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        if self.average_at == "before_activation":
            outputs = _time_averaging(outputs, self.tau, self.states)
            self.states = outputs  # state is integrated input here

        if self.activation is not None:
            outputs = self.activation(outputs)

        if self.average_at == "after_activation":
            outputs = _time_averaging(outputs, self.tau, self.states)
            self.states = outputs  # state is activation here

        return outputs

    def reset_states(self) -> None:
        self.states = None

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "tau": self.tau,
                "average_at": self.average_at,
                "units": self.units,
                "activation": tf.keras.activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "bias_initializer": tf.keras.initializers.serialize(
                    self.bias_initializer
                ),
                "bias_regularizer": tf.keras.regularizers.serialize(
                    self.bias_regularizer
                ),
                "bias_constraint": tf.keras.constraints.serialize(self.bias_constraint),
            }
        )
        return config


class TimeAveragedRNNCell(tf.keras.layers.Layer):
    def __init__(self, tau: float, units: int) -> None:
        super().__init__()
        self.tau = tau
        self.units = units

    def build(self, input_shape: tf.TensorShape) -> None:
        self.input_dense = tf.keras.layers.Dense(self.units, use_bias=False)  # w_xh
        self.recurrent_dense = tf.keras.layers.Dense(self.units, use_bias=False)  # w_hh
        self.time_averaging = MultiInputTimeAveraging(
            tau=self.tau, average_at="after_activation", activation="sigmoid"
        )  # "time-averaging mechanism" with multiple inputs
        self.built = True

    def call(self, inputs: tf.Tensor, states=None) -> Tuple[tf.Tensor, tf.Tensor]:
        xh = self.input_dense(inputs)  # x @ w_xh

        if states is None:
            hh = tf.zeros_like(xh)
        else:
            hh = self.recurrent_dense(states)  # h @ w_hh

        outputs = self.time_averaging(
            [xh, hh]
        )  # sigmoid (tau * (xh + hh + bias) + (1 - tau) * last activation)
        return (
            outputs,
            outputs,
        )  # Consistent with the RNN API, one for state and one for output

    def reset_states(self) -> None:
        self.time_averaging.reset_states()  # Reset the states of the time-averaging mechanism (last activation = None)


class TimeAveragedRNN(tf.keras.layers.Layer):
    def __init__(self, tau: float, units: int) -> None:
        super().__init__()
        self.tau = tau
        self.units = units

    def build(self, input_shape: tf.TensorShape) -> None:
        self.rnn_cell = TimeAveragedRNNCell(tau=self.tau, units=self.units)
        self.built = True

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        max_ticks = inputs.shape[1]  # (batch_size, seq_len, input_dim)
        outputs = tf.TensorArray(dtype=tf.float32, size=max_ticks)
        states = None

        for t in range(max_ticks):
            this_tick_input = inputs[:, t, :]
            states, output = self.rnn_cell(this_tick_input, states=states)
            outputs = outputs.write(t, output)

        # rnn_cell states persist across ticks, but not across batches/calls, so we need to reset rnn_cell here
        self.rnn_cell.reset_states()
        return reshape_proper(outputs)


class PMSPCell(tf.keras.layers.Layer):
    """RNN cell for PMSP model.

    See Plaut, McClelland, Seidenberg and Patterson (1996), simulation 3.
    """

    def __init__(
        self,
        tau: float,
        h_units: int,
        p_units: int,
        c_units: int,
        h_noise: float = 0.0,
        p_noise: float = 0.0,
        c_noise: float = 0.0,
        connections: List[str] = None,
        zero_out_rates: Dict[str, float] = None,
        l2: float = 0.0,
    ) -> None:

        super().__init__()
        self.tau = tau
        self.h_units = h_units
        self.p_units = p_units
        self.c_units = c_units

        [self._validate_noise(x) for x in [h_noise, p_noise, c_noise]]
        self.h_noise = h_noise
        self.p_noise = p_noise
        self.c_noise = c_noise

        if connections is not None:
            self._validate_connections(connections)
        else:
            connections = ["oh", "ph", "hp", "pp", "cp", "pc"]

        self.connections = connections

        if zero_out_rates is not None:
            self._validate_connections(zero_out_rates.keys())
            self.zero_out_rates = zero_out_rates
        else:
            self.zero_out_rates = {c: 0.0 for c in self.connections}

        self.l2 = l2

    @property
    def all_layers_names(self) -> List[str]:
        return [
            "hidden",
            "phonology",
            "cleanup",
        ]  # Prefixes ('h', 'p', 'c') must be unique

    @staticmethod
    def _validate_connections(connections) -> None:
        s = set([letter for connection in connections for letter in connection])

        if not s.issubset(set("ohpc")):
            raise ValueError(
                "Connections must contain only letters in ['o', 'h', 'p', 'c']"
            )

    @staticmethod
    def _validate_noise(noise: float) -> None:
        if noise < 0.0:
            raise ValueError("Noise must be > 0")

    def get_connection_units(self, connection: str) -> int:

        self._validate_connections([connection])
        target = connection[-1]
        return {
            "h": self.h_units,
            "p": self.p_units,
            "c": self.c_units,
        }[target]

    def build(self, input_shape: tf.TensorShape) -> None:

        regularizer = tf.keras.regularizers.L2(self.l2)

        for connection in self.connections:
            setattr(
                self,
                connection,
                ZeroOutDense(
                    units=self.get_connection_units(connection),
                    use_bias=False,
                    name=connection,
                    kernel_regularizer=regularizer,
                    bias_regularizer=regularizer,
                    zero_out_rate=self.zero_out_rates.get(connection, 0.0),
                ),
            )

        # layer block: Add bias, time-averaging and activation
        create_layer = partial(
            MultiInputTimeAveraging,
            tau=self.tau,
            average_at="after_activation",
            activation="sigmoid",
            bias_regularizer=regularizer,  # Only have bias in MITA layer
        )

        for layer in self.all_layers_names:
            setattr(self, layer, create_layer(name=layer))

        # Noise layers (in dict for easier access)
        def _get_noise(layer_name: str) -> float:
            return getattr(self, f"{layer_name[0]}_noise")

        self.noise = {}
        for layer_name in self.all_layers_names:
            self.noise[layer_name] = tf.keras.layers.GaussianNoise(
                _get_noise(layer_name)
            )

        self.built = True

    def get_connections(self, layer: str) -> List[str]:
        """Get connections that end with the given layer.

        e.g., if layer = "hidden", return all connections that ends with "h", e.g.: ["oh", "ph"]
        """
        return [conn for conn in self.connections if conn.endswith(layer[0])]

    def _has_connection(self, layer: str) -> bool:
        """Check whether the given layer has any incoming connections."""
        return len(self.get_connections(layer)) > 0

    def call(
        self,
        last_o: tf.Tensor,
        last_h: tf.Tensor,
        last_p: tf.Tensor,
        last_c: tf.Tensor,
        training: bool = False,
        return_internals: bool = False,
    ) -> Dict[str, tf.Tensor]:
        def get_input(connection) -> tf.Tensor:
            input_map = {
                "o": last_o,
                "h": last_h,
                "p": last_p,
                "c": last_c,
            }
            return input_map[connection[0]]

        batch_size = last_o.shape[0]
        layer_activations = {}
        inputs_to = {}
        for layer_name in self.all_layers_names:
            inputs_to[layer_name] = {}  # Layer inputs, e.g.: noise, x @ w_{xy}

            # Add noise (I did not separate 0 noise from non-zero noise for simplicity, and it is computationally cheap anyway)
            # Do NOT rely on shape broadcasting, since each value must have a random noise value
            _zeros = tf.zeros((batch_size, getattr(self, f"{layer_name[0]}_units")))

            # I use `f"{layer_name}_noise"` instead of `noise` because `input_to` will be flattened at the end
            # Training is always True, because we want to add noise in both training and inference
            inputs_to[layer_name][f"{layer_name}_noise"] = self.noise[layer_name](
                _zeros, training=True
            )

            # Append x @ w for each incoming connection
            if self._has_connection(layer_name):
                for conn in self.get_connections(layer_name):
                    inputs_to[layer_name][conn] = getattr(self, conn)(get_input(conn))

            # Layer activation (bias is inside the MultiInputTimeAveraging layer, so we don't need to add it here)
            layer_activations[layer_name] = getattr(self, layer_name)(
                inputs_to[layer_name].values()
            )

        if return_internals:
            return {
                **layer_activations,
                **inputs_to["hidden"],
                **inputs_to["phonology"],
                **inputs_to["cleanup"],
            }

        return layer_activations

    def reset_states(self):
        """Reset time averaging history."""

        for layer in self.all_layers_names:
            getattr(self, layer).reset_states()

    def zero_out_weights(self):
        """Zero out weights of all layers."""

        for conn in self.connections:
            getattr(self, conn).zero_out_weights()


class PMSPLayer(tf.keras.layers.Layer):
    """PMSP sim 3 model's layer.

    See Plaut, McClelland, Seidenberg and Patterson (1996), simulation 3.
    """

    def __init__(
        self,
        tau: float,
        h_units: int,
        p_units: int,
        c_units: int,
        h_noise: float = 0.0,
        p_noise: float = 0.0,
        c_noise: float = 0.0,
        connections: List[str] = None,
        zero_out_rates: Dict[str, float] = None,
        l2: float = 0.0,
    ) -> None:
        super().__init__()

        # Sanitation is done when building the cell to avoid duplication
        self.tau = tau
        self.h_units, self.p_units, self.c_units = h_units, p_units, c_units
        self.h_noise, self.p_noise, self.c_noise = h_noise, p_noise, c_noise
        self.connections = connections
        self.zero_out_rates = zero_out_rates
        self.l2 = l2

    def build(self, input_shape: tf.TensorShape) -> None:
        self.cell = PMSPCell(
            tau=self.tau,
            h_units=self.h_units,
            p_units=self.p_units,
            c_units=self.c_units,
            h_noise=self.h_noise,
            p_noise=self.p_noise,
            c_noise=self.c_noise,
            connections=self.connections,
            zero_out_rates=self.zero_out_rates,
            l2=self.l2,
        )

        # Lifting `all_layers_name` from cell to layer for easier access
        self.connections = self.cell.connections  # Lift instantiated connections
        self.all_layers_names = self.cell.all_layers_names

        self.built = True

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False,
        return_internals: bool = False,
    ) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:

        batch_size, max_ticks, _ = inputs.shape

        output_names = (
            [*self.all_layers_names, *self.connections]
            if return_internals
            else ["phonology"]
        )

        # Containers for outputs with shape (batch_size, max_ticks, units)
        tf_arrays = {
            name: tf.TensorArray(dtype=tf.float32, size=max_ticks)
            for name in output_names
        }

        # Initialize layer activations
        h = tf.zeros((batch_size, self.h_units))
        p = tf.zeros((batch_size, self.p_units))
        c = tf.zeros((batch_size, self.c_units))

        # Run RNN (Unrolling RNN Cell)
        for t in range(max_ticks):
            o = tf.cast(inputs[:, t], tf.float32)

            cell_outputs = self.cell(
                last_o=o,
                last_h=h,
                last_p=p,
                last_c=c,
                training=training,
                return_internals=return_internals,
            )
            h, p, c = (
                cell_outputs["hidden"],
                cell_outputs["phonology"],
                cell_outputs["cleanup"],
            )

            # Store output arrays
            for name in output_names:
                tf_arrays[name] = tf_arrays[name].write(t, cell_outputs[name])

        self.cell.reset_states()

        return {name: reshape_proper(arr) for name, arr in tf_arrays.items()}


class Spoke(tf.keras.layers.Layer):
    """A spoke in the hub-and-spokes model."""

    def __init__(
        self,
        tau: float,
        units: int,
        average_at="after_activation",
        activation="sigmoid",
        **kwargs,
    ) -> None:
        super().__init__()
        self.tau = tau
        self.units = units  # Begin explicit to cater for `None` input (which is possible in the hub-and-spokes model)
        self.time_averaging = MultiInputTimeAveraging(
            tau=self.tau, average_at=average_at, activation=activation, **kwargs
        )

    def call(
        self, inputs: tf.Tensor = None, cross_tick_states: List[tf.Tensor] = None
    ) -> tf.Tensor:  # to avoid confusing with `self.time_averaging.states` (a_{t-1}), I use a new name `cross_tick_states` here to represent the cross ticks connection.
        """Call the spoke.

        Args:
            inputs (tf.Tensor): clamped inputs to spoke, default to None.
            cross_tick_states (List[tf.Tensor]): states from cross ticks (t-1) projection, a_i w_{ij}, default to None.
        """
        if isinstance(cross_tick_states, list):
            if len(cross_tick_states) == 0:
                cross_tick_states = None

        if inputs is None and cross_tick_states is None:
            return self.time_averaging([tf.zeros((1, self.units))])

        net_inputs = []

        if inputs is not None:
            net_inputs.append(inputs)

        if cross_tick_states is not None:
            net_inputs.extend(cross_tick_states)

        return self.time_averaging(net_inputs)

    def reset_states(self):
        self.time_averaging.reset_states()


class HNSCell(tf.keras.layers.Layer):
    def __init__(
        self,
        tau: float,
        hub_name: str,
        hub_units: int,
        spoke_names: List[str],
        spoke_units: List[int],
    ) -> None:
        super().__init__()
        self.tau = tau
        self.hub_name = hub_name
        self.hub_units = hub_units
        self.spoke_names = spoke_names
        self.spoke_units = spoke_units

    def build(self, input_shape) -> None:

        # Hub
        self.hub = MultiInputTimeAveraging(
            tau=self.tau,
            average_at="after_activation",
            activation="sigmoid",
            use_bias=True,
            name=self.hub_name,
        )
        self.w_hh = self.add_weight(
            shape=(self.hub_units, self.hub_units),
            initializer="random_normal",
            trainable=True,
            name="w_hh",
        )

        # Spokes
        for i, (name, units) in enumerate(zip(self.spoke_names, self.spoke_units)):
            setattr(self, name, Spoke(tau=self.tau, units=units, name=name))

            # Outgoing connection weights (w_sih)
            setattr(
                self,
                f"w_s{i}h",
                self.add_weight(
                    shape=(units, self.hub_units),
                    initializer="random_normal",
                    trainable=True,
                    name=f"w_s{i}h",
                ),
            )

            # Self connection weights (w_sisi)
            setattr(
                self,
                f"w_s{i}s{i}",
                self.add_weight(
                    shape=(units, units),
                    initializer="random_normal",
                    trainable=True,
                    name=f"w_s{i}s{i}",
                ),
            )

            # Incoming connection weights (w_hsi)
            setattr(
                self,
                f"w_hs{i}",
                self.add_weight(
                    shape=(self.hub_units, units),
                    initializer="random_normal",
                    trainable=True,
                    name=f"w_hs{i}",
                ),
            )

        self.built = True

    def _validate_spoke_x(
        self, x: Optional[Dict[str, tf.Tensor]]
    ) -> Dict[str, Optional[tf.Tensor]]:
        """Validate and clean inputs and last_act_spokes."""

        if x is None:
            return {name: None for name in self.spoke_names}

        for name in self.spoke_names:
            if name not in x.keys():
                x[name] = None

        for name in x.keys():
            if name not in self.spoke_names:
                raise ValueError(
                    f"{name} is not one of the spoke name: {self.spoke_names}."
                )

        return x

    def call(
        self,
        inputs: Optional[Dict[str, tf.Tensor]] = None,
        last_act_hub: tf.Tensor = None,
        last_act_spokes: Optional[Dict[str, tf.Tensor]] = None,
        return_internals: bool = False,
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Call the HNSCell.

        Args:
            inputs (dict): clamped inputs with spoke names as keys.
            last_act_hub (tf.Tensor): last activation of the hub.
            last_act_spokes (dict): last activations of the spokes with spoke names as keys.
            return_internals (bool): whether to return the internal states of the spokes. (Default: False)

        Returns:
            If `return_internals` is False:
                act_hub (tf.Tensor): activation of the hub.
                act_spokes (dict): activations of the spokes with spoke names as keys.

            If `return_internals` is True:

        """

        inputs = self._validate_spoke_x(inputs)
        last_act_spokes = self._validate_spoke_x(last_act_spokes)

        layer_activations = {}
        inputs_to = {}

        # Calculate spokes and hub activations
        inputs_to_hub = []
        for i, name in enumerate(self.spoke_names):

            cross_tick_states = []

            if last_act_hub is not None:  # From Hub
                h2s = last_act_hub @ getattr(self, f"w_hs{i}")
                inputs_to[f"{self.hub_name}2{name}"] = h2s
                cross_tick_states.append(h2s)

            if last_act_spokes[name] is not None:  # From self-connection
                s2s = last_act_spokes[name] @ getattr(self, f"w_s{i}s{i}")
                inputs_to[f"{name}2{name}"] = s2s
                cross_tick_states.append(s2s)

            layer_activations[name] = getattr(self, name)(
                inputs[name], cross_tick_states
            )

            # Also append the inputs to hub
            s2h = layer_activations[name] @ getattr(self, f"w_s{i}h")
            inputs_to[f"{name}2{self.hub_name}"] = s2h
            inputs_to_hub.append(s2h)

        # calculate hub activation
        if last_act_hub is not None:
            h2h = last_act_hub @ self.w_hh
            inputs_to[f"{self.hub_name}2{self.hub_name}"] = h2h
            inputs_to_hub.append(h2h)

        layer_activations[self.hub_name] = self.hub(inputs_to_hub)

        # return act_hub, act_spokes

        if return_internals:
            return {**layer_activations, **inputs_to}

        return layer_activations

    @property
    def internals_names(self) -> List[str]:
        """Return the names of the internal connections in a cell."""

        all_layer_names = self.spoke_names + [self.hub_name]
        h2s = [f"{self.hub_name}2{name}" for name in self.spoke_names]
        s2h = [f"{name}2{self.hub_name}" for name in self.spoke_names]
        auto = [f"{name}2{name}" for name in all_layer_names]
        return [*h2s, *s2h, *auto]

    def reset_states(self) -> None:
        self.hub.reset_states()
        for name in self.spoke_names:
            getattr(self, name).reset_states()


class HNSLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        tau: float,
        hub_name: str,
        hub_units: int,
        spoke_names: List[str],
        spoke_units: List[int],
    ) -> None:
        super().__init__()
        self.tau = tau
        self.hub_name, self.hub_units = hub_name, hub_units
        self.spoke_names, self.spoke_units = spoke_names, spoke_units

    def build(self, input_shape) -> None:
        self.cell = HNSCell(
            tau=self.tau,
            hub_name=self.hub_name,
            hub_units=self.hub_units,
            spoke_names=self.spoke_names,
            spoke_units=self.spoke_units,
        )
        self.built = True

    @staticmethod
    def _get_batch_size_and_max_tick(inputs: Dict[str, tf.Tensor]) -> Tuple[int]:
        """Get the batch size and max ticks from inputs."""
        for x in inputs.values():
            if x is not None:
                return x.shape[0], x.shape[1]
        raise ValueError("No input is given, cannot infer batch size or max ticks.")

    def call(
        self, inputs: Dict[str, tf.Tensor], return_internals: bool = False
    ) -> Dict[str, tf.Tensor]:

        inputs = self.cell._validate_spoke_x(inputs)
        batch_size, max_ticks = self._get_batch_size_and_max_tick(inputs)

        # Make containers for outputs
        all_names = [self.hub_name, *self.spoke_names]

        if return_internals:
            all_names.extend(self.cell.internals_names)

        tf_arrays = {
            name: tf.TensorArray(tf.float32, size=max_ticks) for name in all_names
        }

        # Initialize activations in hub and spokes
        hub = tf.zeros((batch_size, self.hub_units))
        spokes = {}
        for name, units in zip(self.spoke_names, self.spoke_units):
            spokes[name] = tf.zeros((batch_size, units))

        # Unrolling
        for t in range(max_ticks):
            y = self.cell(
                inputs={name: v[:, t] for name, v in inputs.items() if v is not None},
                last_act_hub=hub,
                last_act_spokes=spokes,
                return_internals=return_internals,
            )

            for name, arr in tf_arrays.items():
                tf_arrays[name] = arr.write(t, y[name])

            # Overwrite hub and spokes for next tick
            hub = y[self.hub_name]
            spokes = {name: y[name] for name in self.spoke_names}

        # Clear time-averaging states
        self.cell.reset_states()

        return {name: reshape_proper(arr) for name, arr in tf_arrays.items()}
