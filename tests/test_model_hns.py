from hypothesis import given, settings, assume
from hypothesis import strategies as st

from connectionist.models import HubAndSpokes
from connectionist.losses import MaskedBinaryCrossEntropy
import tensorflow as tf


@st.composite
def draw_batch_size(draw):
    batch_size = draw(st.integers(min_value=1, max_value=20))
    assume(20 % batch_size == 0)
    return batch_size


@st.composite
def acceptable_names(draw):
    name = draw(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz")
    )  # no numbers, tf don't like it
    assume(name != "")  # no empty name
    return name


@st.composite
def draw_hs(draw):

    ticks = draw(st.integers(min_value=1, max_value=100))

    # Hub-related
    hub_name = draw(acceptable_names())
    hub_units = draw(st.integers(min_value=1, max_value=100))

    # Spoke-related
    n = draw(st.integers(min_value=1, max_value=10))

    spoke_names = draw(st.lists(acceptable_names(), min_size=n, max_size=n))
    assume(len(set(spoke_names)) == len(spoke_names))  # no duplicates spoke name

    spoke_units = draw(
        st.lists(st.integers(min_value=1, max_value=100), min_size=n, max_size=n)
    )

    # Create training data
    m = draw(st.integers(min_value=1, max_value=n))

    x_train = {}
    for i in range(m):
        x_train[spoke_names[i]] = tf.random.uniform(
            (20, ticks, spoke_units[i]), dtype=tf.float32
        )

    y_train = {hub_name: tf.random.uniform((20, ticks, hub_units), dtype=tf.float32)}

    return hub_name, hub_units, spoke_names, spoke_units, x_train, y_train


@given(
    tau=st.floats(min_value=0.0, max_value=1.0),
    hs=draw_hs(),
    batch_size=draw_batch_size(),
)
def test_training(tau, hs, batch_size):
    """Test the training in HNS model."""

    hub_name, hub_units, spoke_names, spoke_units, x_train, y_train = hs

    model = HubAndSpokes(
        tau=tau,
        hub_name=hub_name,
        hub_units=hub_units,
        spoke_names=spoke_names,
        spoke_units=spoke_units,
    )

    model.compile(
        optimizer="adam",
        loss=MaskedBinaryCrossEntropy(),
    )

    model.fit(x_train, y_train, epochs=3, batch_size=batch_size)
