"""Public argument boundaries should not silently discard invalid values."""

import pytest

from examples.models import normal
from smallprob import ModelError


@pytest.mark.parametrize("value", [[], 0, False, ""])
def test_falsy_non_mapping_inputs_are_rejected(value):
    m = normal(2)
    with pytest.raises(ModelError, match="mapping"):
        m.infer(observed=value)
    with pytest.raises(ModelError, match="mapping"):
        m.simulate(fixed=value)


def test_named_observations_match_mapping_api():
    m = normal(2)
    named = m.infer(y=[1.0, 2.0])
    mapped = m.infer(observed={"y": [1.0, 2.0]})
    assert named.mean("mu") == mapped.mean("mu")
    assert named.sd("mu") == mapped.sd("mu")
    assert named.approximate is False
    assert named.elbo is None


def test_named_and_mapping_observations_cannot_overlap():
    m = normal(2)
    with pytest.raises(ModelError, match="twice"):
        m.infer(observed={"y": [1.0, 2.0]}, y=[3.0, 4.0])
    with pytest.raises(ModelError, match="(overlap|more than once)"):
        m.infer(observed={"y[0]": 1.0}, y=[1.0, 2.0])
