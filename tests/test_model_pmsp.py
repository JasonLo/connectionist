from itertools import product

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from connectionist.data import ToyOP
from connectionist.models import PMSP

DATA = ToyOP()


@st.composite
def draw_conn_rate(draw):
    available_connections = [f"{x[0]}{x[1]}" for x in product("hpc", "hpc")]
    available_connections.extend(["oh", "op", "oc"])
    connections = draw(
        st.lists(st.sampled_from(available_connections), min_size=1, max_size=12)
    )

    # At least a full cycle of the network
    assume(set(["oh", "hp", "pc", "cp"]).issubset(connections))

    zero_out_rates = {
        c: draw(st.floats(min_value=0.0, max_value=1.0)) for c in connections
    }

    return zero_out_rates


@given(
    tau=st.floats(min_value=0.0, max_value=1.0),
    h_units=st.integers(min_value=1, max_value=100),
    p_units=st.just(9),
    c_units=st.integers(min_value=1, max_value=100),
    h_noise=st.floats(min_value=0.0, max_value=1.0),
    p_noise=st.floats(min_value=0.0, max_value=1.0),
    c_noise=st.floats(min_value=0.0, max_value=1.0),
    zero_out_rate=draw_conn_rate(),
    batch_size=st.sampled_from([1, 2, 4, 5, 10, 20]),
)
def test_training(
    tau, h_units, p_units, c_units, h_noise, p_noise, c_noise, zero_out_rate, batch_size
):
    """Test the forward pass in PMSP."""

    model = PMSP(
        tau=tau,
        h_units=h_units,
        p_units=p_units,
        c_units=c_units,
        h_noise=h_noise,
        p_noise=p_noise,
        c_noise=c_noise,
        zero_out_rates=zero_out_rate,
        connections=list(zero_out_rate.keys()),
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
    )

    model.fit(DATA.x_train, DATA.y_train, epochs=5, batch_size=batch_size)
