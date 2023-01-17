import tensorflow as tf


class MaskedBinaryCrossEntropy(tf.keras.losses.Loss):
    """Compute Binary Cross-Entropy with mask.

    Args:
        y_true: target y with shape (batch_size, seq_len, feature)
        y_pred: predicted y with shape (batch_size, seq_len, feature)
        mask_value: value in y_true to be masked
    Returns:
        Loss value with shape (batch_size)
    """

    def __init__(
        self,
        mask_value: int = None,
        name="masked_binary_crossentropy",
        reduction="none",
        **kwargs
    ) -> None:
        super().__init__(name=name, reduction=reduction, **kwargs)
        self.mask_value = mask_value

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
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
