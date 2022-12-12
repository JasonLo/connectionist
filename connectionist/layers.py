from typing import List, Optional
import tensorflow as tf


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

    def get_config(self):
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
