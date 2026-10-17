"""Inspectable Rosenbrock solvers with Armijo, strong Wolfe and trust regions."""
import numpy as np


def objective(z):
    x, y = z
    return float((1-x)**2 + 100*(y-x*x)**2)


def gradient(z):
    x, y = z
    return np.array([2*(x-1)-400*x*(y-x*x), 200*(y-x*x)])


def hessian(z):
    x, y = z
    return np.array([[2-400*y+1200*x*x, -400*x], [-400*x, 200.]])


def strong_wolfe(x, f, g, p, counts, c1=1e-4, c2=.9):
    """Bracket then bisect (zoom); return only steps satisfying both conditions.

    Trial gradients are evaluated only after sufficient decrease. Reuse the
    accepted gradient in the optimizer; all actual evaluations are counted.
    """
    slope = float(g @ p)
    if not slope < 0 or not 0 < c1 < c2 < 1:
        return None

    def evaluate(alpha):
        with np.errstate(over='ignore', invalid='ignore'):
            value = objective(x+alpha*p)
        counts['function'] += 1
        return value

    def derivative(alpha):
        with np.errstate(over='ignore', invalid='ignore'):
            trial_g = gradient(x+alpha*p)
        counts['gradient'] += 1
        return trial_g, float(trial_g @ p)

    def zoom(low, high, low_value):
        for _ in range(60):
            alpha = (low+high)/2
            if alpha == low or alpha == high:
                return None
            value = evaluate(alpha)
            if not np.isfinite(value) or value > f+c1*alpha*slope or value >= low_value:
                high = alpha
            else:
                trial_g, trial_slope = derivative(alpha)
                if not np.isfinite(trial_slope):
                    high = alpha
                    continue
                if abs(trial_slope) <= -c2*slope:
                    return alpha, value, trial_g
                if trial_slope*(high-low) >= 0:
                    high = low
                low, low_value = alpha, value
        return None

    previous, previous_value, alpha = 0., f, 1.
    for i in range(30):
        value = evaluate(alpha)
        if not np.isfinite(value) or value > f+c1*alpha*slope or (i > 0 and value >= previous_value):
            return zoom(previous, alpha, previous_value)
        trial_g, trial_slope = derivative(alpha)
        if not np.isfinite(trial_slope):
            return zoom(previous, alpha, previous_value)
        if abs(trial_slope) <= -c2*slope:
            return alpha, value, trial_g
        if trial_slope >= 0:
            return zoom(alpha, previous, value)
        previous, previous_value = alpha, value
        alpha *= 2
    return None


def dogleg_step(g, H, radius):
    """Dogleg for positive definite models; Cauchy step otherwise."""
    norm = np.linalg.norm(g)
    if norm == 0:
        return np.zeros_like(g)
    curvature = float(g @ H @ g)
    cauchy = -(radius/norm)*g if curvature <= 0 else -min(norm**2/curvature, radius/norm)*g
    if np.linalg.eigvalsh(H)[0] <= 0:
        return cauchy
    newton = np.linalg.solve(H, -g)
    if np.linalg.norm(newton) <= radius:
        return newton
    if np.linalg.norm(cauchy) >= radius*(1-1e-12):
        return cauchy
    direction = newton-cauchy
    a = float(direction @ direction)
    b = float(2*cauchy @ direction)
    c = float(cauchy @ cauchy-radius**2)
    tau = (-b+np.sqrt(max(0., b*b-4*a*c)))/(2*a)
    return cauchy+tau*direction


def trust_region(start, tolerance, max_iterations, initial_radius):
    x = np.asarray(start, dtype=float).copy()
    radius = float(initial_radius)
    f, g = objective(x), gradient(x)
    counts = dict(function=1, gradient=1, hessian=0)
    history = [dict(x=x.tolist(), f=f, norm=float(np.linalg.norm(g)), alpha=None,
                    radius=radius, rho=None)]
    rejected, trials, status = 0, 0, 'Iteration limit'
    H = None
    # Budget counts trial steps, including rejections, to bound total work.
    for _ in range(max_iterations):
        if np.linalg.norm(g) <= tolerance:
            status = 'Converged'
            break
        if H is None:
            H = hessian(x)
            counts['hessian'] += 1
        used_radius = radius
        p = dogleg_step(g, H, radius)
        predicted = -float(g @ p + .5*p @ H @ p)
        if predicted <= 0 or not np.isfinite(predicted):
            status = 'Model reduction failed'
            break
        candidate = x+p
        with np.errstate(over='ignore', invalid='ignore'):
            value = objective(candidate)
        counts['function'] += 1
        trials += 1
        rho = (f-value)/predicted if np.isfinite(value) else -np.inf
        if rho < .25:
            radius *= .25
        elif rho > .75 and np.linalg.norm(p) >= .99*radius:
            radius = min(2*radius, 100.)
        if rho > .1:
            if np.array_equal(x, candidate):
                status = 'Numerical stagnation'
                break
            x, f, g = candidate, value, gradient(candidate)
            counts['gradient'] += 1
            history.append(dict(x=x.tolist(), f=f, norm=float(np.linalg.norm(g)),
                                alpha=None, radius=used_radius, rho=float(rho)))
            H = None
        else:
            rejected += 1
        if radius < 1e-14:
            status = 'Trust radius too small'
            break
    if np.linalg.norm(g) <= tolerance:
        status = 'Converged'
    return dict(method='Trust region', history=history, status=status, counts=counts,
                rejected=rejected, trials=trials)


def nelder_mead(start, max_iterations, simplex_size=.1, simplex_tolerance=1e-6):
    """Classical 2D Nelder–Mead; no derivatives, including in termination."""
    simplex = np.array([start, start+[simplex_size, 0], start+[0, simplex_size]], dtype=float)
    counts = dict(function=0, gradient=0, hessian=0)

    def evaluate(x):
        counts['function'] += 1
        with np.errstate(over='ignore', invalid='ignore'):
            value = objective(x)
        return value if np.isfinite(value) else np.inf

    values = np.array([evaluate(x) for x in simplex])

    def snapshot(operation):
        nonlocal simplex, values
        order = np.argsort(values, kind='stable')
        simplex, values = simplex[order], values[order]
        return dict(x=simplex[0].tolist(), f=float(values[0]), norm=None, alpha=None,
                    simplex=simplex.tolist(), operation=operation,
                    diameter=float(max(np.linalg.norm(a-b) for a in simplex for b in simplex)),
                    spread=float(values[-1]-values[0]))

    history = [snapshot('Initial simplex')]

    def converged():
        return history[-1]['diameter'] <= simplex_tolerance and history[-1]['spread'] <= simplex_tolerance

    for _ in range(max_iterations):
        if converged():
            break
        centroid = simplex[:2].mean(axis=0)
        reflected = centroid+(centroid-simplex[-1])
        reflected_value = evaluate(reflected)
        if reflected_value < values[0]:
            expanded = centroid+2*(reflected-centroid)
            expanded_value = evaluate(expanded)
            if expanded_value < reflected_value:
                simplex[-1], values[-1], operation = expanded, expanded_value, 'Expansion'
            else:
                simplex[-1], values[-1], operation = reflected, reflected_value, 'Reflection'
        elif reflected_value < values[1]:
            simplex[-1], values[-1], operation = reflected, reflected_value, 'Reflection'
        else:
            outside = reflected_value < values[-1]
            contracted = centroid+.5*((reflected if outside else simplex[-1])-centroid)
            contracted_value = evaluate(contracted)
            if (outside and contracted_value <= reflected_value) or (not outside and contracted_value < values[-1]):
                simplex[-1], values[-1], operation = contracted, contracted_value, 'Outside contraction' if outside else 'Inside contraction'
            else:
                simplex[1:] = simplex[0]+.5*(simplex[1:]-simplex[0])
                values[1:] = [evaluate(x) for x in simplex[1:]]
                operation = 'Shrink'
        history.append(snapshot(operation))
    return dict(method='Nelder–Mead', start=np.asarray(start).tolist(), history=history,
                status='Simplex converged' if converged() else 'Iteration limit', counts=counts)


def solve(method, start=(-1.2, 1.), tolerance=1e-6, max_iterations=10000,
          armijo=1e-4, contraction=0.5, initial_radius=1., bfgs_search='Strong Wolfe', wolfe_c2=.9,
          simplex_size=.1, simplex_tolerance=1e-6):
    if method not in ('Gradient descent', 'Newton', 'BFGS', 'Trust region', 'Nelder–Mead'):
        raise ValueError('Unknown method')
    x = np.asarray(start, dtype=float)
    if x.shape != (2,) or not np.all(np.isfinite(x)):
        raise ValueError('Start must contain two finite numbers')
    if not (0 < tolerance < 1 and 0 < armijo < 1 and 0 < contraction < 1):
        raise ValueError('Invalid solver settings')
    if not 1 <= max_iterations <= 50000:
        raise ValueError('Iteration limit must be between 1 and 50000')
    if method == 'Nelder–Mead':
        if not np.isfinite(simplex_size) or not 0 < simplex_size <= 10:
            raise ValueError('Initial simplex size must be in (0, 10]')
        if not np.isfinite(simplex_tolerance) or not 0 < simplex_tolerance < 1:
            raise ValueError('Simplex tolerance must be in (0, 1)')
        return nelder_mead(x, max_iterations, simplex_size, simplex_tolerance)
    if not np.isfinite(initial_radius) or not 0 < initial_radius <= 100:
        raise ValueError('Initial trust radius must be in (0, 100]')
    if bfgs_search not in ('Strong Wolfe', 'Armijo'):
        raise ValueError('Unknown BFGS line search')
    if method == 'BFGS' and bfgs_search == 'Strong Wolfe' and not armijo < wolfe_c2 < 1:
        raise ValueError('Strong Wolfe requires 0 < c1 < c2 < 1')
    if method == 'Trust region':
        return trust_region(x, tolerance, max_iterations, initial_radius)
    f, g, inverse = objective(x), gradient(x), np.eye(2)
    counts = dict(function=1, gradient=1, hessian=0)
    history = [dict(x=x.tolist(), f=f, norm=float(np.linalg.norm(g)), alpha=0.)]
    status, skipped_updates = 'Iteration limit', 0
    for _ in range(max_iterations):
        if np.linalg.norm(g) <= tolerance:
            status = 'Converged'
            break
        if method == 'Newton':
            H = hessian(x)
            counts['hessian'] += 1
            # Shift indefinite Hessians to obtain a positive definite model.
            shift = max(0., 1e-6-float(np.linalg.eigvalsh(H)[0]))
            p = np.linalg.solve(H + shift*np.eye(2), -g)
        elif method == 'BFGS':
            p = -inverse @ g
        else:
            p = -g
        if not np.all(np.isfinite(p)) or g @ p >= 0:
            p, inverse = -g, np.eye(2)
        if method == 'BFGS' and bfgs_search == 'Strong Wolfe':
            step = strong_wolfe(x, f, g, p, counts, armijo, wolfe_c2)
            if step is None:
                status = 'Strong Wolfe search failed'
                break
            alpha, value, new_g = step
            candidate = x+alpha*p
        else:
            alpha = 1.
            for _ in range(60):
                candidate = x + alpha*p
                with np.errstate(over='ignore', invalid='ignore'):
                    value = objective(candidate)
                counts['function'] += 1
                if np.isfinite(value) and value <= f + armijo*alpha*(g @ p):
                    break
                alpha *= contraction
            else:
                status = 'Line search failed'
                break
            new_g = gradient(candidate)
            counts['gradient'] += 1
        s, y = candidate-x, new_g-g
        if method == 'BFGS':
            curvature = float(y @ s)
            if bfgs_search == 'Strong Wolfe' and (not np.isfinite(curvature) or curvature <= 0):
                status = 'Numerical curvature failure'
                break
            if bfgs_search == 'Strong Wolfe' or (curvature > 1e-12*np.linalg.norm(y)*np.linalg.norm(s) and curvature > 0):
                rho = 1/curvature
                V = np.eye(2)-rho*np.outer(s, y)
                inverse = V @ inverse @ V.T + rho*np.outer(s, s)
            else:
                skipped_updates += 1
        stalled = np.array_equal(candidate, x)
        x, f, g = candidate, value, new_g
        history.append(dict(x=x.tolist(), f=f, norm=float(np.linalg.norm(g)), alpha=alpha))
        if stalled:
            status = 'Numerical stagnation'
            break
    if np.linalg.norm(g) <= tolerance:
        status = 'Converged'
    return dict(method=method, history=history, status=status, counts=counts,
                line_search=bfgs_search if method == 'BFGS' else 'Armijo', skipped_updates=skipped_updates)
