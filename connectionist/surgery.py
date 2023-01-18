import random
from dataclasses import dataclass
from typing import Callable, List
import tensorflow as tf


def check_shapes(donor: tf.keras.Model, recipient: tf.keras.Model) -> None:
    """Check that the shapes of the weights are the same."""
    for w_donor, w_recipient in zip(donor.weights, recipient.weights):
        print(
            f"Checking: {w_donor.name}: {w_donor.shape} -> {w_recipient.name} : {w_recipient.shape}, shape changed: {w_donor.shape == w_recipient.shape}"
        )


def get_suffix(weight_name: str) -> str:
    """Get last 2 segments of a weight name."""
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


@dataclass
class SurgeryPlan:
    """A surgery plan for removing shrink_rate amount of the units in `target_layer`.

    TODO: Support for enlargement surgery?
    """

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
        print(f"Keeping {self.keep_n} out of {self.original_units} original units")
        if self.keep_n == 0:
            raise ValueError(
                f"Shrink rate {self.shrink_rate} is too high, no units left."
            )

        # Generate indices to keep (shared across all weights)
        self.keep_idx = sorted(random.sample(range(self.original_units), self.keep_n))
        print(f"Keep indices are: {self.keep_idx}")

    def __repr__(self):
        return f"SurgeryPlan(target_layer={self.layer}, original_units={self.original_units}, damage={self.shrink_rate}, keep_idx={self.keep_idx}, keep_n={self.keep_n})"


def make_recipient(model, surgery_plan: SurgeryPlan, make_model_fn: Callable):
    """Make a recipient model according to surgery plan."""

    config = model.get_config()

    to_config_key = {
        "hidden": "h_units",
        "phonology": "p_units",
        "cleanup": "c_units",
    }

    config[to_config_key[surgery_plan.layer]] = surgery_plan.keep_n
    print(f"New config: {config}")
    return make_model_fn(**config)


class Surgeon:
    """A class for transplanting weights from one model to another according to surgery_plan.

    Args:
        surgery_plan: A surgery plan for the transplant (specifying where the damage happens).

    The `surgery_plan` specify where the shrinkage happens, it will move all related weights using `lesion_transplant()` method.
    For other weights that are not related to the surgery_plan, it will just copy them over using `copy_transplant()` method.

    Also see connectionist.models.PMSP.shrink_layer().

    Example:

    ```python

    from connectionist.surgery import *
    from connectionist.data import ToyOP

    data = ToyOP()
    donor_model = PMSP(tau=0.2, h_units=10, p_units=9, c_units=5)
    y = donor_model(data.x_train)  # for instantiating the weights

    # Create surgery plan and surgeon
    plan = SurgeryPlan(layer='hidden', original_units=10, shrink_rate=0.5)
    surgeon = Surgeon(surgery_plan=plan)

    # Create recipient model and transplant weights
    new_model = make_recipient(model=donor_model, surgery_plan=plan, make_model_fn=PMSP)
    new_model.build(input_shape=donor_model.pmsp._build_input_shape)
    surgeon.transplant(donor=donor_model, recipient=new_model)
    ```

    """

    def __init__(self, surgery_plan: SurgeryPlan) -> None:
        self.plan = surgery_plan

    @staticmethod
    def _validate_axis(
        w_donor: tf.Tensor,
        w_recipient: tf.Tensor,
        axis: List[int],
    ) -> None:

        if not 0 < len(axis) <= 2:
            raise ValueError(f"Axis must be of length 1 or 2, got {axis}")

        # Check non self-connecting weights shapes
        if len(axis) == 1 and len(w_donor.shape) > 1:
            match_ax = 1 - axis[0]
            if w_donor.shape[match_ax] != w_recipient.shape[match_ax]:
                raise ValueError(
                    f"In {w_donor.name}, shapes don't match on axis {match_ax}: {w_donor.shape=}, {w_recipient.shape=}"
                )

    def lesion_transplant(
        self,
        donor: tf.keras.Model,
        recipient: tf.keras.Model,
        weight_name: str,
        idx: List[int],
        axis: List[int],
    ) -> None:
        """Transplant weights from donor to recipient in the weights that requires shrinking."""

        w_recipient = get_weights(recipient, weight_name)
        w_donor = get_weights(donor, weight_name)
        self._validate_axis(w_donor, w_recipient, axis=axis)

        print(
            f"Transplanting: {w_donor.name}:{w_donor.shape} -> {w_recipient.name}: {w_recipient.shape}"
        )

        w = tf.identity(w_donor)  # Copy the weights
        for a in axis:
            w = tf.gather(
                w, indices=idx, axis=a
            )  # slice the weight in each connected axis
        w_recipient.assign(w)  # assign the weights to recipient

    def transplant(self, donor: tf.keras.Model, recipient: tf.keras.Model) -> None:
        """Transplant all the weights from donor to recipient model."""

        # Execute lesion transplant (Move weights and remove a subset of units)
        conn_locs = donor.connection_locs[self.plan.layer]
        weight_names = list(conn_locs.keys())
        axis = list(conn_locs.values())

        for name, ax in zip(weight_names, axis):
            self.lesion_transplant(
                donor=donor,
                recipient=recipient,
                weight_name=name,
                idx=self.plan.keep_idx,
                axis=ax,
            )

        # Execute simple transplant (Only move weights)
        all_weights_names = list(donor.weights_abbreviations.keys())
        remaining_weights = [
            name for name in all_weights_names if name not in weight_names
        ]
        for name in remaining_weights:
            copy_transplant(donor=donor, recipient=recipient, weight_name=name)
