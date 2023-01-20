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
        try:
            matched = [
                w
                for w in model.weights
                if model.weights_abbreviations[weight_name] in w.name
            ]
        except AttributeError:
            pass

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
    """Transplant all specified weights and biases from `donor` to `recipient`.

    All weights and biases shapes must match.

    Args:
        donor (tf.keras.Model): The donor model.
        recipient (tf.keras.Model): The recipient model.
        weight_name (str): The name of the weights to transplant, can be full internal name,
            partial internal name (will match with .endswith) or abbreviation if `model.weights_abbreviation` exists).

    """

    w_recipient = get_weights(recipient, weight_name)
    w_donor = get_weights(donor, weight_name)

    print(
        f"Transplanting: {w_donor.name}:{w_donor.shape} -> {w_recipient.name}: {w_recipient.shape}"
    )

    if w_donor.shape != w_recipient.shape:
        raise ValueError(
            f"Shapes don't match: {w_donor.shape} != {w_recipient.shape} for {w_donor.name} -> {w_recipient.name}"
        )

    w_recipient.assign(w_donor)


@dataclass
class SurgeryPlan:
    """A surgery plan for shrinking a layer, removing `shrink_rate` amount of the units in `layer`.

    Reducing the units in a layer will also reduce the number units in all connected weights and bias.
    The index of removal is random and shared across all weights and biases.

    Args:
        layer (str): The layer name to shrink. e.g.(hidden, phonology, cleanup in PMSP).
        original_units (int): The original number of units in the layer.
        shrink_rate (float): The shrink rate, between 0 and 1.
        make_model_fn (Callable): A function that make the original and new model.

    """

    layer: str
    original_units: int
    shrink_rate: float
    make_model_fn: Callable

    def __post_init__(self) -> None:
        """Validate plan and random sample the indices of unit to keep."""

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

    def __repr__(self) -> str:
        return f"SurgeryPlan(layer={self.layer}, original_units={self.original_units}, damage={self.shrink_rate}, keep_idx={self.keep_idx}, keep_n={self.keep_n})"


def make_recipient(
    donor: tf.keras.Model, layer: str, keep_n: int, make_model_fn: Callable
) -> tf.keras.Model:
    """Make a recipient model according to donor and surgery plan for shrinking a layer.

    Args:
        donor (tf.keras.Model): The donor model.
        layer (str): The layer name to shrink. e.g.(hidden, phonology, cleanup in PMSP).
        keep_n (int): The number of units to keep.
        make_model_fn (Callable): A function that make the original and new model.

    """

    config = donor.get_config()

    to_config_key = {
        "hidden": "h_units",
        "phonology": "p_units",
        "cleanup": "c_units",
    }

    config[to_config_key[layer]] = keep_n
    print(f"New config: {config}")
    return make_model_fn(**config)


class Surgeon:
    """A surgeon transplanting weights from one model to another according to a [SurgeryPlan][connectionist.surgery.SurgeryPlan].

    The [SurgeryPlan][connectionist.surgery.SurgeryPlan] only specify where the shrinkage happens,
    when calling [transplant][connectionist.surgery.Surgeon.transplant], it will:

    - use [lesion_transplant][connectionist.surgery.Surgeon.lesion_transplant] to copy over the weights that requires shrinking.
    - use [copy_transplant][connectionist.surgery.copy_transplant] on the remaining weights.

    Args:
        surgery_plan (SurgeryPlan): A surgery plan for the transplant.

    !!! Example
        ```python
        import tensorflow as tf
        from connectionist.data import ToyOP
        from connectionist.models import PMSP
        from connectionist.surgery import SurgeryPlan, Surgeon, make_recipient

        # Create model and train
        data = ToyOP()
        model = PMSP(tau=0.2, h_units=10, p_units=9, c_units=5)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(),
        )
        model.fit(data.x_train, data.y_train, epochs=10, batch_size=20)

        # Create surgery plan and surgeon
        plan = SurgeryPlan(layer='hidden', original_units=10, shrink_rate=0.3, make_model_fn=PMSP)
        surgeon = Surgeon(surgery_plan=plan)

        # Create recipient model and transplant weights
        new_model = make_recipient(model=model, layer=plan.layer, keep_n=plan.keep_n, make_model_fn=plan.make_model_fn)
        new_model.build(input_shape=model.pmsp._build_input_shape)

        # Transplant weights and biases
        surgeon.transplant(donor=model, recipient=new_model)
        ```

    Also see [connectionist.models.PMSP.shrink_layer][] for high-level API.
    """

    def __init__(self, surgery_plan: SurgeryPlan) -> None:
        self.plan = surgery_plan

    @staticmethod
    def _validate_axes(
        w_donor: tf.Tensor,
        w_recipient: tf.Tensor,
        axes: List[int],
    ) -> None:

        if not 0 < len(axes) <= 2:
            raise ValueError(f"Axis must be of length 1 or 2, got {axes}")

        # Check non self-connecting weights shapes
        if len(axes) == 1 and len(w_donor.shape) > 1:
            match_ax = 1 - axes[0]
            if w_donor.shape[match_ax] != w_recipient.shape[match_ax]:
                raise ValueError(
                    f"In {w_donor.name}, shapes don't match on axis {match_ax}: {w_donor.shape=}, {w_recipient.shape=}"
                )

    def lesion_transplant(
        self,
        donor: tf.keras.Model,
        recipient: tf.keras.Model,
        weight_name: str,
        keep_idx: List[int],
        axes: List[int],
    ) -> None:
        """Transplant weights from donor to recipient in the weights that requires shrinking.

        Args:
            donor (tf.keras.Model): The donor model.
            recipient (tf.keras.Model): The recipient model.
            weight_name (str): The name of the weights to transplant.
            keep_idx (List[int]): The indices to keep.
            axes (List[int]): The axes to slice the weights, usually contains only 1 axis, but 2 when it is a self-connecting weight.
        """

        w_recipient = get_weights(recipient, weight_name)
        w_donor = get_weights(donor, weight_name)
        self._validate_axes(w_donor, w_recipient, axes=axes)

        print(
            f"Transplanting: {w_donor.name}:{w_donor.shape} -> {w_recipient.name}: {w_recipient.shape}"
        )

        w = tf.identity(w_donor)  # Copy the weights
        for a in axes:
            w = tf.gather(
                w, indices=keep_idx, axis=a
            )  # slice the weight in each connected axis
        w_recipient.assign(w)  # assign the weights to recipient

    def transplant(self, donor: tf.keras.Model, recipient: tf.keras.Model) -> None:
        """Transplant all weights from donor to recipient model.

        Args:
            donor (tf.keras.Model): The donor model.
            recipient (tf.keras.Model): The recipient model.

        """

        # Execute lesion transplant (Move weights and remove a subset of units)
        conn_locs = donor._connection_locs[self.plan.layer]
        weight_names = list(conn_locs.keys())
        axis = list(conn_locs.values())

        for name, ax in zip(weight_names, axis):
            self.lesion_transplant(
                donor=donor,
                recipient=recipient,
                weight_name=name,
                keep_idx=self.plan.keep_idx,
                axes=ax,
            )

        # Execute simple copy transplant in the remaining weights
        all_weights_names = list(donor.weights_abbreviations.keys())
        remaining_weights = [
            name for name in all_weights_names if name not in weight_names
        ]
        for name in remaining_weights:
            copy_transplant(donor=donor, recipient=recipient, weight_name=name)
