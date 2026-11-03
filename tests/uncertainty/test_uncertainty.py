import copy

import numpy as np
import pytest
import torch

from driveaware.data.corruptions import gaussian_noise, motion_blur, occlusion
from driveaware.evaluation.uncertainty_metrics import (
    coverage,
    fit_sigma_scale,
    gaussian_nll,
    sparsification,
    spearman,
)
from driveaware.models.temporal.gru_policy import TemporalConfig, TemporalPolicy
from driveaware.models.uncertainty.ensemble import (
    EnsemblePredictor,
    TrajectoryUncertainty,
    mc_dropout,
)
from driveaware.training.temporal import laplace_nll


def _prob_model(seed):
    torch.manual_seed(seed)
    return TemporalPolicy(TemporalConfig(probabilistic=True)).eval()


def _inputs(b=3):
    torch.manual_seed(99)
    return torch.randn(b, 4, 256), torch.rand(b, 4) * 8, torch.tensor([0, 1, 2][:b])


def test_probabilistic_head_shapes_and_positive_scale():
    m = _prob_model(0)
    mean, b = m.forward_dist(*_inputs())
    assert mean.shape == (3, 10, 2) and b.shape == (3, 10, 2)
    assert (b >= m.cfg.min_scale).all()
    torch.testing.assert_close(m.forward_features(*_inputs()), mean)


def test_deterministic_model_has_no_dist():
    with pytest.raises(ValueError):
        TemporalPolicy(TemporalConfig()).forward_dist(*_inputs())


def test_laplace_nll_minimised_at_true_scale():
    torch.manual_seed(0)
    target = torch.distributions.Laplace(0.0, 0.5).sample((20000,))
    mean = torch.zeros_like(target)
    vals = {
        s: laplace_nll(mean, torch.full_like(target, s), target).item() for s in (0.25, 0.5, 1.0)
    }
    assert min(vals, key=vals.get) == 0.5


def test_ensemble_variance_decomposition():
    a = _prob_model(1)
    identical = EnsemblePredictor([a, copy.deepcopy(a)]).predict(*_inputs())
    assert np.allclose(identical.epistemic_var, 0.0)
    assert (identical.aleatoric_var > 0).all()
    diverse = EnsemblePredictor([_prob_model(s) for s in range(5)]).predict(*_inputs())
    assert (diverse.epistemic_var > 0).all() and diverse.member_means.shape == (5, 3, 10, 2)
    np.testing.assert_allclose(diverse.total_var, diverse.epistemic_var + diverse.aleatoric_var)
    np.testing.assert_allclose(diverse.mean, diverse.member_means.mean(0), rtol=1e-6)
    s = TrajectoryUncertainty.summary(diverse.total_var)
    assert s.shape == (3,) and (s > 0).all()


def test_mc_dropout_restores_eval_mode():
    m = TemporalPolicy(TemporalConfig()).eval()
    out = mc_dropout(m, *_inputs(), passes=8)
    assert out.member_means.shape == (8, 3, 10, 2) and (out.epistemic_var > 0).any()
    assert not any(mod.training for mod in m.modules())


def test_spearman_and_sparsification():
    rng = np.random.default_rng(0)
    e = rng.random(500)
    assert spearman(e, e) == pytest.approx(1.0)
    assert spearman(e, -e) == pytest.approx(-1.0)
    perfect = sparsification(e, e)
    assert perfect["ause"] == pytest.approx(0.0, abs=1e-12)
    rand = sparsification(e, rng.random(500))
    assert rand["ause"] > 0.1
    assert perfect["method"][0] == pytest.approx(1.0)


def test_gaussian_coverage_nll_and_scale_fit():
    rng = np.random.default_rng(1)
    var = np.full((20000, 2), 4.0)
    err = rng.normal(0, 2.0, (20000, 2))
    assert coverage(err, var, 1.0) == pytest.approx(0.6827, abs=0.01)
    assert coverage(err, var, 2.0) == pytest.approx(0.9545, abs=0.01)
    assert gaussian_nll(err, var) < gaussian_nll(err, var / 10)
    assert fit_sigma_scale(err, var / 4) == pytest.approx(2.0, rel=0.03)  # sigma too small by 2


def test_corruptions_deterministic_and_shape_preserving():
    imgs = np.random.default_rng(2).integers(0, 255, (4, 96, 160, 3), dtype=np.uint8)
    for out in (gaussian_noise(imgs, 20, 0), motion_blur(imgs, 9), occlusion(imgs, 0.25, 0)):
        assert out.shape == imgs.shape and out.dtype == np.uint8 and not np.array_equal(out, imgs)
    np.testing.assert_array_equal(gaussian_noise(imgs, 20, 3), gaussian_noise(imgs, 20, 3))
    occ = occlusion(imgs, 0.25, 0)
    assert np.isclose((occ.reshape(4, -1, 3).sum(-1) == 0).mean(axis=1), 0.25, atol=0.03).all()
