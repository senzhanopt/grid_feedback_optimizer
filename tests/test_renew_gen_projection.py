from grid_feedback_optimizer.engine.renew_gen_projection import RenewGenProjection
import numpy as np
from numpy.testing import assert_allclose

def test_analytic_cvxpy():

    renew_gen_proj = RenewGenProjection()

    p_max, p_min, s_inv, p, q = 4.0, 0.0, 5.0, 5.0, 1.0
    expected = np.array([4.0, 1.0])

    # Run both projections
    result_analytic = renew_gen_proj.analytic_projection(p_max, s_inv, p, q)
    result_opt = renew_gen_proj.opt_projection(p_max, p_min, s_inv, p, q)

    # Check both methods agree
    assert_allclose(result_analytic, expected, atol=1e-6)
    assert_allclose(result_opt, expected, atol=1e-6)
    assert_allclose(result_analytic, result_opt, atol=1e-6)

    p, q = 4.0, 4.0
    expected = np.array([5.0, 5.0])/np.sqrt(2)

    # Run both projections
    result_analytic = renew_gen_proj.analytic_projection(p_max, s_inv, p, q)
    result_opt = renew_gen_proj.opt_projection(p_max, p_min, s_inv, p, q)

    # Check both methods agree
    assert_allclose(result_analytic, expected, atol=1e-6)
    assert_allclose(result_opt, expected, atol=1e-6)
    assert_allclose(result_analytic, result_opt, atol=1e-6)