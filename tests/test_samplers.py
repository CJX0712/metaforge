import numpy as np

from metaforge.data.samplers import GaussianBlobSampler, SinusoidSampler, blob_preset


def test_blob_sizes_and_labels():
    s = GaussianBlobSampler(n_way=5, k_shot=3, q_shot=7, dim=4, class_sep=3.0, sigma=1.0, seed=1)
    t = s.sample()
    assert t.x_support.shape == (15, 4)
    assert t.x_query.shape == (35, 4)
    assert t.y_support.shape == (15,)
    assert sorted(set(t.y_support.tolist())) == list(range(5))
    # every query label appears in support
    assert set(t.y_query.tolist()).issubset(set(t.y_support.tolist()))
    t.check()


def test_blob_determinism():
    a = GaussianBlobSampler(n_way=4, k_shot=2, q_shot=3, dim=3, seed=42).sample()
    b = GaussianBlobSampler(n_way=4, k_shot=2, q_shot=3, dim=3, seed=42).sample()
    assert np.allclose(a.x_support, b.x_support)
    assert np.array_equal(a.y_support, b.y_support)


def test_sinusoid_shapes():
    s = SinusoidSampler(n_support=10, n_query=12, seed=3)
    t = s.sample()
    assert t.x_support.shape == (10, 1)
    assert t.y_support.shape == (10, 1)
    assert t.x_query.shape == (12, 1)
    # sinusoid should be roughly bounded
    assert np.all(np.abs(t.y_support) < 6.0)


def test_preset_levels_distinct():
    easy = blob_preset("easy", seed=1).sample()
    hard = blob_preset("hard", seed=1).sample()
    # different separability -> different support statistics (very unlikely identical)
    assert not np.allclose(easy.x_support, hard.x_support)
