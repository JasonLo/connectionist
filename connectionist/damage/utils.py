import tensorflow as tf


def check_shapes(donor: tf.keras.Model, recipient: tf.keras.Model) -> None:
    """Check that the shapes of the weights are the same."""
    for w_donor, w_recipient in zip(donor.weights, recipient.weights):
        print(
            f"Checking: {w_donor.name}: {w_donor.shape} -> {w_recipient.name} : {w_recipient.shape}, shape changed: {w_donor.shape == w_recipient.shape}"
        )


def get_suffix(weight_name: str) -> str:
    """Get last 2 segnments of a weight name."""
    return "/".join(weight_name.split("/")[-2::])


def get_weights(model: tf.keras.Model, weight_name: str) -> tf.Tensor:
    """Get weights by full name, endswith, or abbreviation."""

    # Full name matching
    matched = [w for w in model.weights if w.name == weight_name]

    # Ends with name matching
    if len(matched) == 0:
        matched = [w for w in model.weights if w.name.endswith(weight_name)]

    # Abbreviation matching
    if len(matched) == 0:
        matched = [
            w
            for w in model.weights
            if model.weights_abbreviations[weight_name] in w.name
        ]

    if len(matched) == 0:
        raise ValueError(f"Could not find weights for {weight_name}")
    if len(matched) > 1:
        raise ValueError(
            f"Found multiple weights: {[w.name for w in matched]} with similar name to {weight_name}, please be more specific."
        )
    return matched[0]


def copy_transplant(
    donor: tf.keras.Model, recipient: tf.keras.Model, weight_name: str
) -> None:
    """Transplant weights from donor to recipient in weights that has matching shape."""

    w_recipient = get_weights(recipient, weight_name)
    w_donor = get_weights(donor, weight_name)

    print(
        f"Transplanting: {w_donor.name}:{w_donor.shape} -> {w_recipient.name}: {w_recipient.shape}"
    )

    assert (
        w_donor.shape == w_recipient.shape
    ), f"Shapes don't match: {w_donor.shape} != {w_recipient.shape}"

    w_recipient.assign(w_donor)
