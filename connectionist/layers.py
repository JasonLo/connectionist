from typing import Dict, List, Optional, Tuple, Union
import tensorflow as tf

# TODO: add support for regularization


def _time_averaging(
    x: tf.Tensor, tau: float, states: Optional[tf.Tensor] = None
) -> tf.Tensor:
    """Time-averaging mechanism."""

    if states is None:
        return x * tau

    return x * tau + (1 - tau) * states


class TimeAveragedDense(tf.keras.layers.Dense):
    """Dense layer with Time-averaging mechanism.

    In short, time-averaging mechanism simulates continuous-temporal dynamics in a discrete-time recurrent neural networks.
    See Plaut, McClelland, Seidenberg, and Patterson (1996) equation (15) for more details.

    Args:
        tau (float): Time-averaging parameter (How much information should take from the new input). range: [0, 1].

        average_at (str): Where to average. Options: 'before_activation', 'after_activation'.

            When average_at is 'before_activation', the time-averaging is applied BEFORE activation. i.e., time-averaging INPUT.:
                outputs = activation(integrated_input);
                integrated input = tau * (inputs @ weights + bias) + (1-tau) * last_inputs;
                last_inputs is obtained from the last call of this layer, its values stored at `self.states`

            When average_at is 'after_activation', the time-averaging is applied AFTER activation. i.e., time-averaging OUTPUT.:
                outputs = tau * activation(inputs @ weights + bias) + (1-tau) * last_outputs;
                last_outputs is obtained from the last call of this layer, its values stored at `self.states`

        kwargs (str, optional): Any argument in keras.layers.Dense. (e.g., units, activation, use_bias, kernel_initializer, bias_initializer,
            kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, **kwargs).

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
            self.states = outputs

        if self._activation is not None:
            outputs = self._activation(outputs)

        if self.average_at == "after_activation":
            outputs = _time_averaging(outputs, self.tau, self.states)
            self.states = outputs

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


class MultiInputTimeAveraging(tf.keras.layers.Layer):
    """Time-averaging mechanism for multiple inputs.

    In short, time-averaging mechanism simulates continuous-temporal dynamics in a discrete-time recurrent neural networks.
    See Plaut, McClelland, Seidenberg, and Patterson (1996) equation (15) for more details.

    This layer is designed for multiple inputs, assuming they had ALREADY been multiplied by weights i.e., a list of (x @ w).

    Args:
        tau (float): Time-averaging parameter (How much information should take from the new input). range: [0, 1].

        average_at (str): Where to average. Options: 'before_activation', 'after_activation'.

            When average_at is 'before_activation', the time-averaging is applied BEFORE activation. i.e., time-averaging INPUT.:
                outputs = activation(integrated_input);
                integrated input = tau * (sum(inputs) + bias) + (1-tau) * last_inputs;
                last_inputs is obtained from the last call of this layer, its values stored at `self.states`

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

        outputs = self.sum(inputs)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        if self.average_at == "before_activation":
            outputs = _time_averaging(outputs, self.tau, self.states)
            self.states = outputs

        if self.activation is not None:
            outputs = self.activation(outputs)

        if self.average_at == "after_activation":
            outputs = _time_averaging(outputs, self.tau, self.states)
            self.states = outputs

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

    def reset_states(self) -> None:  # TODO: maybe need another name
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
        outputs = outputs.stack()  # (seq_len, batch_size, units)
        outputs = tf.transpose(outputs, [1, 0, 2])  # (batch_size, seq_len, units)
        return outputs


class PMSPCell(tf.keras.layers.Layer):
    """RNN cell for PMSP model.

    See Plaut, McClelland, Seidenberg and Patterson (1996), simulation 3.
    """

    def __init__(self, tau: float, h_units: int, p_units: int, c_units: int) -> None:
        super().__init__()
        self.tau = tau
        self.h_units = h_units
        self.p_units = p_units
        self.c_units = c_units

    def build(self, input_shape: tf.TensorShape) -> None:
        # Hidden layer
        self.o2h = tf.keras.layers.Dense(
            self.h_units, activation=None, use_bias=False, name="o2h"
        )  # w_oh
        self.p2h = tf.keras.layers.Dense(
            self.h_units, activation=None, use_bias=False, name="p2h"
        )  # w_ph
        self.time_averaging_h = MultiInputTimeAveraging(
            tau=self.tau,
            average_at="after_activation",
            activation="sigmoid",
            name="ta_h",
        )  # bias_h and the time averaging mechanism

        # Phonology layer
        self.h2p = tf.keras.layers.Dense(
            self.p_units, activation=None, use_bias=False, name="h2p"
        )  # w_hp
        self.p2p = tf.keras.layers.Dense(
            self.p_units, activation=None, use_bias=False, name="p2p"
        )  # w_pp
        self.c2p = tf.keras.layers.Dense(
            self.p_units, activation=None, use_bias=False, name="c2p"
        )  # w_cp
        self.time_averaging_p = MultiInputTimeAveraging(
            tau=self.tau,
            average_at="after_activation",
            activation="sigmoid",
            name="ta_p",
        )  # bias_p and the time averaging mechanism

        # Cleanup layer
        self.p2c = TimeAveragedDense(
            tau=self.tau,
            average_at="after_activation",
            units=self.c_units,
            activation="sigmoid",
            name="p2c",
        )  # w_pc, bias_c, and the time averaging mechanism
        self.built = True

    def call(
        self,
        last_o: tf.Tensor,
        last_h: tf.Tensor,
        last_p: tf.Tensor,
        last_c: tf.Tensor,
        return_internals: bool = False,
    ) -> Dict[str, tf.Tensor]:
        # Hidden layer activation
        # h_t = tau(act(o_{t-1} @ w_oh + p_{t-1} @ w_ph + bias_h)) + (1 - tau) * h_{t-1}
        oh = self.o2h(last_o)
        ph = self.p2h(last_p)
        h = self.time_averaging_h([oh, ph])

        # Phonology layer activation
        # p_t = tau(act(h_{t-1} @ w_hp + p_{t-1} @ w_pp + c_{t-1} @ w_cp + bias_p)) + (1 - tau) * p_{t-1}
        hp = self.h2p(last_h)
        pp = self.p2p(last_p)
        cp = self.c2p(last_c)
        p = self.time_averaging_p([hp, pp, cp])

        # Cleanup layer activation
        # c_t = tau(act(p_{t-1} @ w_pc + bias_c)) + (1 - tau) * c_{t-1}
        c = self.p2c(last_p)

        if return_internals:
            return {
                "h": h,
                "p": p,
                "c": c,
                "oh": oh,
                "ph": ph,
                "hp": hp,
                "pp": pp,
                "cp": cp,
            }

        return {"h": h, "p": p, "c": c}

    def reset_states(self):  # TODO: need another name?
        """Reset time averaging history."""
        self.time_averaging_p.reset_states()
        self.time_averaging_h.reset_states()
        self.p2c.reset_states()


class PMSPLayer(tf.keras.layers.Layer):
    """PMSP sim 3 model's layer.

    See Plaut, McClelland, Seidenberg and Patterson (1996), simulation 3.
    """

    def __init__(self, tau: float, h_units: int, p_units: int, c_units: int) -> None:
        super().__init__()
        self.tau = tau
        self.h_units = h_units
        self.p_units = p_units
        self.c_units = c_units

    def build(self, input_shape: tf.TensorShape) -> None:
        self.cell = PMSPCell(
            tau=self.tau,
            h_units=self.h_units,
            p_units=self.p_units,
            c_units=self.c_units,
        )

        self.built = True

    def call(
        self, inputs: tf.Tensor, return_internals: bool = False
    ) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:

        batch_size, max_ticks, _ = inputs.shape

        names = (
            ["h", "p", "c", "oh", "ph", "hp", "pp", "cp"] if return_internals else ["p"]
        )

        # Containers for outputs with shape (batch_size, max_ticks, units)
        tf_arrays = {
            name: tf.TensorArray(dtype=tf.float32, size=max_ticks) for name in names
        }

        h = tf.zeros((batch_size, self.h_units))
        p = tf.zeros((batch_size, self.p_units))
        c = tf.zeros((batch_size, self.c_units))

        # Run RNN (Unrolling RNN Cell)
        for t in range(max_ticks):
            o = inputs[:, t]

            cell_outputs = self.cell(
                last_o=o,
                last_h=h,
                last_p=p,
                last_c=c,
                return_internals=return_internals,
            )
            h, p, c = cell_outputs["h"], cell_outputs["p"], cell_outputs["c"]

            # Store output arrays
            for name in names:
                tf_arrays[name] = tf_arrays[name].write(t, cell_outputs[name])

        self.cell.reset_states()

        # Stack and transpose output arrays
        for name in names:
            tf_arrays[name] = tf_arrays[name].stack()
            tf_arrays[name] = tf.transpose(tf_arrays[name], [1, 0, 2])

        if return_internals:
            return tf_arrays
        else:
            return tf_arrays["p"]
