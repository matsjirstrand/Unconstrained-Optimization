# Rosenbrock optimization lab

A local Dash application comparing gradient descent, safeguarded Newton, BFGS, trust-region Newton and derivative-free Nelder–Mead on
`f(x,y) = (1-x)^2 + 100(y-x^2)^2`.

## Run

With Python 3.11 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python app.py
```

Open http://127.0.0.1:8050. The initial experiment runs automatically. Choose a preset,
edit coordinates or click the contour plot, then press **Run experiment**. Select
methods, tolerance and iteration budget in the sidebar. Use playback controls or
drag the iteration slider to inspect paths. CSV export includes every accepted
iterate, objective, gradient norm and step length.

Use **Stop app** in the sidebar to terminate the local server process. The page
stays open but becomes inactive. Restart with `./run.ps1` from PowerShell in the
project folder (or run `python app.py` in your configured environment).

## Numerical conventions

- Derivative-based solvers stop when the Euclidean gradient norm is at most the chosen tolerance.
- Nelder–Mead uses reflection (1), expansion (2), contraction (0.5) and shrink
  (0.5). The initial simplex is the start plus one coordinate offset per axis,
  with adjustable size (default 0.1). It stops when both the maximum pairwise
  vertex distance and the objective spread meet its separate simplex tolerance.
  This is not a gradient convergence guarantee. It never evaluates derivatives;
  displayed gradient norms are diagnostics computed afterward, excluded from
  solver evaluation counts. Playback shows the best vertex and current triangle
  at each simplex iteration; the best vertex may stay unchanged during an iteration.
  CSV includes simplex diameter, objective spread and operation.
  From the classic start at simplex tolerance 1e-6 it takes 111 iterations and
  205 function evaluations, so the default 100-iteration budget stops it early.
- Gradient descent and Newton use Armijo backtracking, starting at 1 with
  c1=1e-4 and contraction=0.5, with at most 60 trials per iteration.
- Newton shifts the Hessian when its smallest eigenvalue is below 1e-6.
- BFGS defaults to strong Wolfe: Armijo sufficient decrease plus
  `abs(gradient(x+alpha*p) @ p) <= c2 * abs(gradient(x) @ p)`, with c2=0.9.
  The search brackets by doubling from alpha=1 (up to 30 trials), then zooms
  with bisection (up to 60 trials). The accepted gradient is reused. Every
  accepted BFGS step updates the inverse Hessian; numerical nonpositive curvature
  terminates explicitly instead of silently skipping the update.
  The **BFGS line search** selector retains the original Armijo option for comparison,
  including its curvature skip safeguard. Both start from the identity inverse
  Hessian. Non-descent directions fall back to negative gradient.
- Trust region uses a dogleg step for positive definite Hessians and a Cauchy step
  otherwise, using the original Hessian model. The initial radius is adjustable.
  Actual/predicted reduction below 0.25 shrinks the radius by 4; above 0.75 doubles
  it for boundary steps, up to 100. Steps are accepted for a ratio above 0.1.
  Its budget counts all trials, including rejected steps; playback shows accepted
  steps only. Rejections are reported separately. CSV includes the radius used
  and reduction ratio for each accepted step; line-search length is blank.
- Function evaluations include rejected trials. Strong Wolfe also evaluates
  gradients at some rejected trial points, so gradient counts may exceed the
  number of accepted points. Hessians are counted separately.
- The default budget of 100 is insufficient for gradient descent at 1e-6;
  an iteration-limit status is an explicit incomplete solve, not convergence.
- Plot lines connect accepted iterates; they are not continuous optimizer dynamics.
  Contour lines use the exact formula y=x² ± sqrt(c-(1-x)²)/10 at f=c.
  The log10(f) background is resampled over the visible area on zoom, with
  a display-only floor of 1e-20. Smaller levels are added near the optimum.
- Playback reveals one accepted iteration per frame across all three plots by
  default, at two frames per second. **Playback speed** offers one, two or five
  frames per second. Increase **Iterations per frame** to accelerate long runs. **Step +1**
  always advances one iteration. Comparison uses iteration index, not elapsed runtime.

## Verify

```powershell
python -m unittest -v
```

Tests check analytic derivatives against central differences, convergence from
three starting points, monotone descent, stationary starts and iteration limits.
The numerical core in `optimization.py` only depends on NumPy; the interface lives
in `app.py`. Downloaded project-local packages in `.packages` are ignored by Git.

## BFGS comparison

At gradient tolerance 1e-6, c1=1e-4, c2=0.9 and Armijo contraction=0.5:

| Start | Armijo steps | Strong Wolfe steps | Armijo f/g evaluations | Strong Wolfe f/g evaluations |
| --- | ---: | ---: | ---: | ---: |
| (-1.2, 1) | 34 | 34 | 54 / 35 | 55 / 36 |
| (0, 1) | 26 | 25 | 42 / 27 | 41 / 27 |
| (2, 2) | 47 | 41 | 78 / 48 | 68 / 46 |

All converge. Armijo skipped zero updates in these three runs; the differences
come from the accepted step lengths. Strong Wolfe does not improve every start.
