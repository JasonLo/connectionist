import tensorflow as tf


class MaskedBinaryCrossEntropy(tf.keras.losses.Loss):
    """Compute Binary Cross-Entropy with masking.

    Args:
        mask_value (int, optional): value in y_true to be masked.
        name (str, optional): name of the loss function.
        reduction (str, optional): reduction method for the loss.
    """

    def __init__(
        self,
        mask_value: int = None,
        name: str = "masked_binary_crossentropy",
        reduction: str = "none",
        **kwargs
    ) -> None:
        super().__init__(name=name, reduction=reduction, **kwargs)
        self.mask_value = mask_value

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        r"""Calculate masked binary cross-entropy.

        Define as:
        ```math
        H_p(q) = - \frac{1}{M} \sum_{i=1}^{N} m_i \cdot (y_i \cdot \log(p(y_i)) + (1-y_i) \cdot \log(1-p(y_i)))
        ```
        where $`y_i`$ is the target, and $`p(y_i)`$ is the prediction $`m_i`$ is the mask, and $`M`$ is the number of unmasked units.

        !!! warning
            "$`\cdot`$" is element-wise multiplication.

        Args:
            y_true (tf.Tensor): target y with shape (batch_size, seq_len, feature)
            y_pred (tf.Tensor): predicted y with shape (batch_size, seq_len, feature)

        Returns:
            Loss (tf.Tensor): Loss values with shape (batch_size)

        """

        epsilon = tf.keras.backend.epsilon()  # very small value to avoid log(0)
        cross_entropy = y_true * tf.math.log(y_pred + epsilon)
        cross_entropy = cross_entropy + (1 - y_true) * tf.math.log(1 - y_pred + epsilon)

        if self.mask_value:
            mask = tf.cast(
                tf.where(y_true == self.mask_value, 0, 1), tf.float32
            )  # create mask
        else:
            mask = tf.ones_like(cross_entropy)  # All inclusive mask if value is none

        cross_entropy = mask * cross_entropy  # zero out the masked values
        cross_entropy = tf.reduce_sum(
            cross_entropy, axis=[1, 2]
        )  # sum over all units (axis 2) and time steps (axis 1)
        return -cross_entropy / (
            epsilon + tf.reduce_sum(mask, axis=[1, 2])
        )  # - (1/N) sum(y * log(p(y)) + (1-y) * log(1-p(y)))
