import numpy as np
from numpy.testing import assert_allclose

from grid_feedback_optimizer.engine.renew_gen_projection import RenewGenProjection


def test_analytic_cvxpy():

    renew_gen_proj = RenewGenProjection()

    p_max, p_min, s_inv, p, q = 4.0, 0.0, 5.0, 5.0, 1.0
    expected = np.array([4.0, 1.0])
    result = renew_gen_proj.projection(p_max=p_max, p_min=p_min, p=p, q=q, s_inv=s_inv)
    assert_allclose(result, expected, atol=1e-6)

    p, q = 4.0, 4.0
    expected = np.array([5.0, 5.0]) / np.sqrt(2)
    result = renew_gen_proj.projection(p_max=p_max, p_min=p_min, p=p, q=q, s_inv=s_inv)
    assert_allclose(result, expected, atol=1e-6)

    result = renew_gen_proj.projection(
        p_max=p_max, p_min=p_min, p=p, q=q, s_inv=s_inv, pf_min=0.9
    )
