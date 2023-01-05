from typing import List, Callable
import random
from dataclasses import dataclass
import tensorflow as tf


def check_shapes(donor: tf.keras.Model, recipient: tf.keras.Model) -> None:
    """Check that the shapes of the weights are the same."""
    for w_donor, w_recipient in zip(donor.weights, recipient.weights):
        print(
            f"Checking: {w_donor.name}: {w_donor.shape} -> {w_recipient.name} : {w_recipient.shape}, shape changed: {w_donor.shape == w_recipient.shape}"
        )


@dataclass
class SurgeryPlan:
    """A surgery plan for removing `damage` percent of the units in `target_layer`."""

    layer: str
    original_units: int
    shrink_rate: float

    def __post_init__(self):
        """Validate the surgery plan."""

        if not (0 < self.shrink_rate < 1):
            raise ValueError(
                f"Shrink rate must be between 0 and 1, got {self.shrink_rate}"
            )

        # Keeping `keep_n` units
        self.keep_n = int(self.original_units * (1 - self.shrink_rate))
        print(f"Keeping {self.keep_n} out of {self.original_units} orginal units")
        if self.keep_n == 0:
            raise ValueError(
                f"Shrink rate {self.shrink_rate} is too high, no units left."
            )

        # Generate indices to keep (shared across all weigths)
        self.keep_idx = sorted(random.sample(range(self.original_units), self.keep_n))
        print(f"Keep indices are: {self.keep_idx}")

    def __repr__(self):
        return f"SurgeryPlan(target_layer={self.layer}, original_units={self.original_units}, damage={self.shrink_rate}, keep_idx={self.keep_idx}, keep_n={self.keep_n})"


def make_recipient(model, surgery_plan: SurgeryPlan, make_model_fn: Callable):
    """Make a recipient model according to surgery plan."""

    config = model.get_config()

    layer_to_config_key = {
        "hidden": "h_units",
        "phonology": "p_units",
        "cleanup": "c_units",
    }

    k = layer_to_config_key[surgery_plan.layer]
    config[k] = surgery_plan.keep_n
    print(f"New config: {config}")
    return make_model_fn(**config)


def get_weights(model: tf.keras.Model, name: str) -> tf.Tensor:
    """Get weights by name abbreviation.

    Important: it is a partial matching by the name (using name in weight.name).
    """

    matched = [w for w in model.weights if name in w.name]
    assert (
        len(matched) == 1
    ), f"Found {len(matched)} weights for {name}, make sure the name is unique enough."
    return matched[0]


class Surgeon:
    """A class for transplanting weights from one model to another."""

    def __init__(self, surgery_plan: SurgeryPlan) -> None:
        self.plan = surgery_plan

    def lesion_transplant(
        self,
        donor: tf.keras.Model,
        recipient: tf.keras.Model,
        name: str,
        idx: List[int],
        axis: int,
    ) -> None:
        """Transplant weights from donor to recipient."""

        w_recipient = get_weights(recipient, name)
        w_donor = get_weights(donor, name)

        if len(w_donor.shape) > 1:  # Check weights only, not biases
            should_match_ax = 1 - axis
            assert (
                w_donor.shape[should_match_ax] == w_recipient.shape[should_match_ax]
            ), f"Shapes don't match at axis {should_match_ax}: {w_donor.shape} != {w_recipient.shape}"

        print(
            f"Transplanting: {w_donor.name}:{w_donor.shape} -> {w_recipient.name}: {w_recipient.shape}"
        )
        w_recipient.assign(tf.gather(w_donor, indices=idx, axis=axis))

    def simple_transplant(
        self, donor: tf.keras.Model, recipient: tf.keras.Model, name: str
    ) -> None:
        """Transplant weights from donor to recipient."""

        w_recipient = get_weights(recipient, name)
        w_donor = get_weights(donor, name)

        assert (
            w_donor.shape == w_recipient.shape
        ), f"Shapes don't match: {w_donor.shape} != {w_recipient.shape}"

        print(
            f"Transplanting: {w_donor.name}:{w_donor.shape} -> {w_recipient.name}: {w_recipient.shape}"
        )
        w_recipient.assign(w_donor)

    def transplant(self, donor: tf.keras.Model, recipient: tf.keras.Model) -> None:
        """Transplant all the weights from donor to recipient model."""

        # Execute lesion transplant (Move weights and remove a subset of units)
        names, axis = donor.layer2weights(self.plan.layer)

        for name, ax in zip(names, axis):
            self.lesion_transplant(
                donor=donor,
                recipient=recipient,
                name=name,
                idx=self.plan.keep_idx,
                axis=ax,
            )

        # Execute simple transplant (Only move weights)
        all_weights_names = list(donor.abbreviations.values())
        remaining_weights = [name for name in all_weights_names if name not in names]
        for name in remaining_weights:
            self.simple_transplant(donor=donor, recipient=recipient, name=name)
