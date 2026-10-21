"""Typed probability tables may contain roundoff, but never arbitrary weights."""
import numpy as np
import pytest

from smallprob import Categorical, HMM, ModelError, Normal


def test_float32_roundoff_uses_one_normalized_law():
    raw = np.array([.85, .15], dtype=np.float32)
    law = Categorical(raw)
    probabilities = law._probabilities({})
    expected = raw.astype(float) / raw.astype(float).sum()
    np.testing.assert_array_equal(probabilities, expected)
    assert sum(np.exp(law.log_prob(k, {})) for k in range(2)) == pytest.approx(1, abs=1e-15)
    assert law.sample({}, np.random.default_rng(7)) in [0, 1]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_meaningfully_unnormalized_weights_still_rejected(dtype):
    for values in ([.85, .14], [2., 3.], [0., 0.], [-.1, 1.1]):
        with pytest.raises(ModelError):
            Categorical(np.array(values, dtype=dtype))


def test_hmm_float32_tables_match_normalized_reference():
    initial = np.array([.85, .15], dtype=np.float32)
    transition = np.array([[.85, .15], [.1, .9]], dtype=np.float32)
    chain = HMM(initial=initial, transition=transition, emissions=Normal([-1., 1.], .7), steps=3)
    ref_initial = initial.astype(float)
    ref_initial /= ref_initial.sum()
    ref_transition = transition.astype(float)
    ref_transition /= ref_transition.sum(axis=1, keepdims=True)
    reference = HMM(initial=ref_initial, transition=ref_transition, emissions=Normal([-1., 1.], .7), steps=3)
    p, ref = chain.infer(y=[-.4, .2, 1.]), reference.infer(y=[-.4, .2, 1.])
    np.testing.assert_allclose(p.probabilities('z'), ref.probabilities('z'), rtol=1e-14, atol=1e-14)
    assert p.log_evidence == pytest.approx(ref.log_evidence, abs=1e-14)
    torch = pytest.importorskip('torch')
    native = chain.simulate(seed=3, backend='torch', dtype='float64')
    assert isinstance(native['z'], torch.Tensor)
