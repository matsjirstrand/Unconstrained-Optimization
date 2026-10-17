import unittest
from unittest.mock import patch
import numpy as np
from optimization import objective, gradient, hessian, solve, dogleg_step, strong_wolfe


class SolverTests(unittest.TestCase):
    def test_nelder_mead_derivative_free_convergence(self):
        for start in [(-1.2, 1), (0, 1), (2, 2)]:
            with self.subTest(start=start), patch('optimization.gradient', side_effect=AssertionError('Derivative used')), patch('optimization.hessian', side_effect=AssertionError('Hessian used')), patch('optimization.objective', wraps=objective) as evaluations:
                r = solve('Nelder–Mead', start, max_iterations=500)
                self.assertEqual(r['status'], 'Simplex converged')
                self.assertEqual(r['counts']['function'], evaluations.call_count)
                self.assertEqual(r['counts']['gradient'], 0)
                self.assertEqual(r['counts']['hessian'], 0)
                last = r['history'][-1]
                self.assertLessEqual(last['diameter'], 1e-6)
                self.assertLessEqual(last['spread'], 1e-6)
                np.testing.assert_allclose(last['x'], [1, 1], atol=1e-6)
                self.assertTrue(np.all(np.diff([p['f'] for p in r['history']]) <= 0))
                self.assertTrue(all(np.array(p['simplex']).shape == (3, 2) for p in r['history']))

    def test_nelder_mead_budget_shrink_and_validation(self):
        r = solve('Nelder–Mead', max_iterations=1)
        self.assertEqual(r['status'], 'Iteration limit')
        self.assertEqual(len(r['history']), 2)
        with patch('optimization.objective', return_value=1):
            r = solve('Nelder–Mead', max_iterations=1)
        self.assertEqual(r['history'][-1]['operation'], 'Shrink')
        self.assertAlmostEqual(r['history'][-1]['diameter'], r['history'][0]['diameter']/2)
        self.assertEqual(r['counts']['function'], 7)
        for size in [0, -1, float('nan')]:
            with self.assertRaises(ValueError):
                solve('Nelder–Mead', simplex_size=size)

    def test_derivatives(self):
        for x in [np.array([-1.2, 1.]), np.array([0., 1.]), np.array([1.3, -.7])]:
            eps = 1e-5
            basis = np.eye(2)*eps
            numeric_g = np.array([(objective(x+d)-objective(x-d))/(2*eps) for d in basis])
            numeric_H = np.column_stack([(gradient(x+d)-gradient(x-d))/(2*eps) for d in basis])
            np.testing.assert_allclose(gradient(x), numeric_g, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(hessian(x), numeric_H, rtol=1e-6, atol=1e-6)

    def test_convergence_and_descent(self):
        for method in ['Gradient descent', 'Newton', 'BFGS', 'Trust region']:
            for start in [(-1.2, 1), (0, 1), (2, 2)]:
                with self.subTest(method=method, start=start):
                    result = solve(method, start, max_iterations=30000)
                    self.assertEqual(result['status'], 'Converged')
                    self.assertLessEqual(result['history'][-1]['norm'], 1e-6)
                    np.testing.assert_allclose(result['history'][-1]['x'], [1, 1], atol=3e-6)
                    f = [p['f'] for p in result['history']]
                    self.assertTrue(np.all(np.diff(f) <= 1e-14))
                    self.assertGreaterEqual(result['counts']['gradient'], len(f))

    def test_bfgs_wolfe_conditions(self):
        for start in [(-1.2, 1), (0, 1), (2, 2)]:
            r = solve('BFGS', start)
            self.assertEqual(r['status'], 'Converged')
            self.assertEqual(r['skipped_updates'], 0)
            for before, after in zip(r['history'], r['history'][1:]):
                x, next_x = np.array(before['x']), np.array(after['x'])
                s = next_x-x
                p = s/after['alpha']
                slope = gradient(x) @ p
                self.assertLessEqual(after['f'], before['f']+1e-4*after['alpha']*slope+1e-14)
                self.assertLessEqual(abs(gradient(next_x) @ p), .9*abs(slope)+1e-13)
                self.assertGreater((gradient(next_x)-gradient(x)) @ s, 0)
            old = solve('BFGS', start, bfgs_search='Armijo')
            self.assertEqual(old['status'], 'Converged')

    def test_wolfe_expands_short_steps(self):
        x = np.array([-1.2, 1.])
        g = gradient(x)
        p = -1e-6*g
        counts = dict(function=0, gradient=0)
        alpha, f, next_g = strong_wolfe(x, objective(x), g, p, counts)
        self.assertGreater(alpha, 1)
        self.assertLessEqual(f, objective(x)+1e-4*alpha*(g @ p))
        self.assertLessEqual(abs(next_g @ p), .9*abs(g @ p))

    def test_stationary_start_and_limit(self):
        self.assertEqual(len(solve('Newton', (1, 1))['history']), 1)
        self.assertEqual(solve('BFGS', max_iterations=1)['status'], 'Iteration limit')

    def test_trust_region_model_and_rejections(self):
        for H in [np.diag([2., 5.]), np.diag([-2., 5.])]:
            for radius in [.01, 1., 10.]:
                g = np.array([2., -3.])
                p = dogleg_step(g, H, radius)
                self.assertLessEqual(np.linalg.norm(p), radius*(1+1e-12))
                self.assertLess(g @ p + .5*p @ H @ p, 0)
        r = solve('Trust region', initial_radius=10.)
        self.assertEqual(r['status'], 'Converged')
        self.assertGreater(r['rejected'], 0)
        self.assertEqual(r['trials'], len(r['history'])-1+r['rejected'])
        self.assertEqual(r['counts']['function'], r['trials']+1)
        for previous, current in zip(r['history'], r['history'][1:]):
            self.assertLessEqual(np.linalg.norm(np.array(current['x'])-previous['x']), current['radius']*(1+1e-12))
            self.assertGreater(current['rho'], .1)
        self.assertEqual(solve('Trust region', (1, 1))['status'], 'Converged')


if __name__ == '__main__':
    unittest.main()
