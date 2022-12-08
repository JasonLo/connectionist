import tensorflow as tf


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
            outputs = self._time_averaging(outputs)

        if self._activation is not None:
            outputs = self._activation(outputs)

        if self.average_at == "after_activation":
            outputs = self._time_averaging(outputs)

        return outputs

    def _time_averaging(self, x: tf.Tensor) -> tf.Tensor:
        """Time-averaging mechanism."""
        x *= self.tau

        if self.states is not None:
            x += self.states * (1 - self.tau)

        self.states = tf.constant(x)
        return x

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
