from typing import List, Callable
from dataclasses import dataclass
import random
import tensorflow as tf
from .utils import get_weights, copy_transplant


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

    to_config_key = {
        "hidden": "h_units",
        "phonology": "p_units",
        "cleanup": "c_units",
    }

    config[to_config_key[surgery_plan.layer]] = surgery_plan.keep_n
    print(f"New config: {config}")
    return make_model_fn(**config)


class Surgeon:
    """A class for transplanting weights from one model to another according to sugery_plan.

    Args:
        surgery_plan: A surgery plan for the transplant (specifying where the damage happens).

    The `surgery_plan` specify where the shrinkage happens, it will move all related weights using `lesion_transplant()` method.
    For other weights that are not related to the surgery_plan, it will just copy them over using `copy_transplant()` method.

    Also see connectionist.models.PMSP.shrink_layer().

    Example:

    ```python

    from connectionist.damage.shrink_layer import *
    from connectionist.data import ToyOP

    data = ToyOP()
    donor_model = PMSP(tau=0.2, h_units=10, p_units=9, c_units=5)
    y = donor_model(data.x_train)  # for instantiating the weights

    # Create surgery plan and surgeon
    plan = SurgeryPlan(layer='hidden', original_units=10, shrink_rate=0.5)
    surgeon = Surgeon(surgery_plan=plan)

    # Create receipient model and transplant weights
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
