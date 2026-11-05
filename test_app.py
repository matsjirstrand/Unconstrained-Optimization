"""Integration checks for the app and its selected Python environment."""
import csv
import io
from pathlib import Path
import sys
import unittest

import dash
import numpy
import plotly
import app


class AppTests(unittest.TestCase):
    def test_dependencies_use_interpreter_environment(self):
        environment = Path(sys.prefix).resolve()
        for module in (numpy, dash, plotly):
            with self.subTest(module=module.__name__):
                self.assertTrue(Path(module.__file__).resolve().is_relative_to(environment))
        legacy = Path(__file__).parent / '.packages'
        self.assertNotIn(str(legacy), sys.path)

    def test_routes_and_assets(self):
        client = app.app.server.test_client()
        for route in ('/', '/_dash-layout', '/_dash-dependencies', '/assets/playback.js', '/assets/style.css'):
            with self.subTest(route=route):
                with client.get(route) as response:
                    self.assertEqual(response.status_code, 200)

    def test_experiment_playback_zoom_and_export(self):
        runs, error = app.experiment(None, -1.2, 1, app.METHODS, 1e-6, 150, 1e-4, .5)
        self.assertEqual(error, '')
        self.assertEqual(len(runs), 5)
        for result in runs:
            if result['method'] == 'Gradient descent':
                self.assertEqual(result['status'], 'Iteration limit')
            else:
                self.assertIn('converged', result['status'].lower())
        self.assertEqual(runs[-1]['counts']['gradient'], 0)
        viewport = {'xaxis.range[0]': .99, 'xaxis.range[1]': 1.01,
                    'yaxis.range[0]': .98, 'yaxis.range[1]': 1.02}
        for iteration in (0, 5, 150):
            contour, values, gradients, _, _ = app.render(runs, iteration, viewport)
            self.assertEqual(tuple(contour.layout.xaxis.range), (.99, 1.01))
            self.assertEqual(tuple(contour.layout.yaxis.range), (.98, 1.02))
            for i, result in enumerate(runs):
                expected = min(iteration+1, len(result['history']))
                self.assertEqual(len(values.data[2*i].x), expected)
                self.assertEqual(len(gradients.data[2*i].x), expected)
            # Exercise Plotly's serializer with the current NumPy/Python wheels.
            self.assertIn('data', contour.to_json())
        rows = list(csv.DictReader(io.StringIO(app.export(1, runs)['content'])))
        self.assertEqual(len(rows), sum(len(r['history']) for r in runs))
        self.assertIn('simplex_diameter', rows[-1])


if __name__ == '__main__':
    unittest.main()
