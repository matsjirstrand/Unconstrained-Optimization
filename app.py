"""Run with python app.py, then open http://127.0.0.1:8050."""
import csv
import io
import os
import threading
import numpy as np
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update, ClientsideFunction
import plotly.graph_objects as go
from optimization import solve, gradient

METHODS = ['Gradient descent', 'Newton', 'BFGS', 'Trust region', 'Nelder–Mead']
COLORS = ['#ef9b35', '#4a9eff', '#28c6a4', '#d59aff', '#f27fa7']
app = Dash(__name__, title='Rosenbrock · Optimization lab')


def field(label, component):
    return html.Label([html.Span(label), component])


app.layout = html.Div([
    html.Header([html.Div('NUMERICAL EXPLORATIONS', className='eyebrow'),
                 html.H1('Inside the banana valley.'),
                 html.P('Five algorithms. One curved valley. Explore how each method finds its way to the minimum.')]),
    html.Div([
        html.Aside([
            html.H3('Experiment controls'),
            field('Starting point preset', dcc.Dropdown(id='preset', options=[
                {'label': 'Classic · (−1.2, 1)', 'value': 'classic'},
                {'label': 'Across the valley · (2, 2)', 'value': 'across'},
                {'label': 'Indefinite Hessian · (0, 1)', 'value': 'indefinite'}], value='classic', clearable=False)),
            html.Div([field('Start x', dcc.Input(id='x', type='number', value=-1.2, min=-5, max=5, step='any')),
                      field('Start y', dcc.Input(id='y', type='number', value=1, min=-5, max=5, step='any'))], className='pair'),
            field('Methods', dcc.Checklist(id='methods', options=METHODS, value=METHODS)),
            field('Gradient tolerance', dcc.Dropdown(id='tol', options=[1e-4, 1e-6, 1e-8], value=1e-6, clearable=False)),
            field('Maximum iterations', dcc.Input(id='limit', type='number', value=100, min=1, max=50000, step=1)),
            field('Initial trust-region radius', dcc.Input(id='radius', type='number', value=1, min=.0001, max=100, step='any')),
            field('Nelder–Mead initial simplex size', dcc.Input(id='simplex-size', type='number', value=.1, min=.000001, max=10, step='any')),
            field('Nelder–Mead simplex tolerance', dcc.Dropdown(id='simplex-tol', options=[1e-4, 1e-6, 1e-8], value=1e-6, clearable=False)),
            field('BFGS line search', dcc.Dropdown(id='bfgs-search', options=['Strong Wolfe', 'Armijo'], value='Strong Wolfe', clearable=False)),
            html.Details([html.Summary('Line-search settings'),
                field('Armijo constant c₁', dcc.Input(id='armijo', type='number', value=.0001, min=.000001, max=.9, step='any')),
                field('Armijo backtracking factor', dcc.Input(id='beta', type='number', value=.5, min=.01, max=.99, step='any')),
                field('Wolfe curvature constant c₂', dcc.Input(id='c2', type='number', value=.9, min=.01, max=.99, step='any'))]),
            html.Button('Run experiment →', id='run', className='primary'),
            html.Div(id='error', role='alert'),
            html.Hr(), html.Div('THE BENCHMARK', className='eyebrow'),
            dcc.Markdown('$$f(x,y)=(1-x)^2+100(y-x^2)^2$$', mathjax=True),
            html.P('Global minimum: f(1, 1) = 0. Click the level-set plot to choose a new starting point.'),
            html.P('Gradient descent and Newton use Armijo backtracking. BFGS defaults to strong Wolfe (sufficient decrease and curvature); choose Armijo to compare. Trust region uses dogleg steps with a Cauchy fallback for indefinite Hessians.', className='muted'),
            html.P('Nelder–Mead uses only objective values. It stops when both simplex diameter and objective spread meet its simplex tolerance. Gradient norms are computed afterward for comparison only, outside the solver evaluation counts.', className='muted'),
            html.Hr(), html.Button('Stop app', id='stop-app', className='danger'),
            html.Div(id='shutdown-status', role='status')
        ]),
        html.Main([
            html.Div([html.H3('Paths to the minimum'), html.Span('Logarithmic level sets', className='badge')], className='panel-title'),
            dcc.Graph(id='landscape', config={'displaylogo': False}),
            html.Div([html.Button('▶ Play', id='play'), html.Button('Step +1', id='step'), html.Button('Show final', id='final'),
                      html.Button('Export CSV', id='export')], className='transport'),
            field('Iterations per frame (1 shows every step)', dcc.Input(id='stride', type='number', value=1, min=1, max=1000, step=1)),
            field('Playback speed', dcc.Dropdown(id='playback-speed', options=[
                {'label': 'Slow · 1 frame / second', 'value': 1000},
                {'label': 'Normal · 2 frames / second', 'value': 500},
                {'label': 'Fast · 5 frames / second', 'value': 200}], value=500, clearable=False)),
            dcc.Slider(id='iteration', min=0, max=1, value=0, step=1, marks=None, tooltip={'placement': 'bottom', 'always_visible': True}),
            html.P(id='frame-label', className='muted'),
            html.Div([dcc.Graph(id='values'), dcc.Graph(id='gradients')], className='charts'),
            html.H3('Results at termination'), html.Div(id='results'),
            html.P('Playback counts accepted steps, or simplex iterations for Nelder–Mead (the best point may repeat). Its current triangle is shown in pink. Completed methods stay at their endpoint. Evaluation counts include rejected trials. Trust-region budgets count all trial steps.', className='muted')
        ])
    ], className='workspace'),
    dcc.Store(id='runs'), dcc.Interval(id='clock', interval=500, disabled=True), dcc.Download(id='download')
], className='shell')


@app.callback(Output('shutdown-status', 'children'), Output('stop-app', 'disabled'),
              Input('stop-app', 'n_clicks'), prevent_initial_call=True)
def stop_app(clicks):
    if not clicks:
        return no_update, no_update
    # Let Dash deliver the status before terminating this server process only.
    timer = threading.Timer(1.0, lambda: os._exit(0))
    timer.daemon = True
    timer.start()
    return 'App stopped. To restart it, run .\\run.ps1 from the project folder.', True


@app.callback(Output('x', 'value'), Output('y', 'value'), Input('preset', 'value'), Input('landscape', 'clickData'), prevent_initial_call=True)
def choose_start(preset, click):
    if ctx.triggered_id == 'landscape' and click:
        p = click['points'][0]
        return round(p['x'], 4), round(p['y'], 4)
    return {'classic': (-1.2, 1), 'across': (2, 2), 'indefinite': (0, 1)}[preset]


@app.callback(Output('runs', 'data'), Output('error', 'children'), Input('run', 'n_clicks'),
              State('x', 'value'), State('y', 'value'), State('methods', 'value'), State('tol', 'value'),
              State('limit', 'value'), State('armijo', 'value'), State('beta', 'value'), State('radius', 'value'),
              State('bfgs-search', 'value'), State('c2', 'value'), State('simplex-size', 'value'), State('simplex-tol', 'value'))
def experiment(_, x, y, methods, tol, limit, armijo, beta, radius=1., bfgs_search='Strong Wolfe', c2=.9,
               simplex_size=.1, simplex_tol=1e-6):
    try:
        if x is None or y is None or not (-5 <= x <= 5 and -5 <= y <= 5):
            raise ValueError('Choose a starting point within [−5, 5] on both axes.')
        if not methods:
            raise ValueError('Select at least one method.')
        if limit is None or int(limit) != limit:
            raise ValueError('Iteration limit must be an integer.')
        runs = [solve(m, (x, y), tol, int(limit), armijo, beta, radius, bfgs_search, c2,
                      simplex_size, simplex_tol) for m in methods]
        # Post-processing diagnostics are never used by the derivative-free solver.
        for r in runs:
            if r['method'] == 'Nelder–Mead':
                for p in r['history']:
                    p['norm'] = float(np.linalg.norm(gradient(p['x'])))
        return runs, ''
    except (ValueError, TypeError) as e:
        return no_update, str(e)


app.clientside_callback(
    ClientsideFunction(namespace='optimizationLab', function_name='playback'),
    Output('iteration', 'max'), Output('iteration', 'value'), Output('clock', 'disabled'), Output('play', 'children'),
    Input('runs', 'data'), Input('play', 'n_clicks'), Input('step', 'n_clicks'),
    Input('final', 'n_clicks'), Input('clock', 'n_intervals'),
    State('iteration', 'value'), State('clock', 'disabled'), State('stride', 'value'))

app.clientside_callback(
    'function (speed) { return speed || 500; }',
    Output('clock', 'interval'), Input('playback-speed', 'value'))


def base(title, x='Iteration', y=''):
    fig = go.Figure()
    fig.update_layout(template='plotly_dark', paper_bgcolor='#111c2c', plot_bgcolor='#111c2c',
                      font=dict(color='#cbd5e1'), title=title, margin=dict(l=55, r=20, t=50, b=45),
                      xaxis_title=x, yaxis_title=y, legend=dict(orientation='h', y=1.12))
    return fig


@app.callback(Output('landscape', 'figure'), Output('values', 'figure'), Output('gradients', 'figure'),
              Output('results', 'children'), Output('frame-label', 'children'),
              Input('runs', 'data'), Input('iteration', 'value'), Input('landscape', 'relayoutData'))
def render(runs, iteration, viewport=None):
    runs = runs or []
    iteration = iteration or 0
    points = [p['x'] for r in runs for p in r['history']] + [[-2, -1], [2, 3]]
    points += [vertex for r in runs for p in r['history'] for vertex in p.get('simplex', [])]
    a = np.array(points)
    low, high = a.min(axis=0)-.25, a.max(axis=0)+.25
    for i, axis in enumerate(('xaxis', 'yaxis')):
        view = viewport or {}
        bounds = view.get(f'{axis}.range')
        if bounds is None and all(f'{axis}.range[{j}]' in view for j in (0, 1)):
            bounds = [view[f'{axis}.range[0]'], view[f'{axis}.range[1]']]
        if bounds is not None and not view.get(f'{axis}.autorange'):
            low[i], high[i] = sorted(bounds)
    x, y = np.linspace(low[0], high[0], 400), np.linspace(low[1], high[1], 400)
    X, Y = np.meshgrid(x, y)
    F = (1-X)**2+100*(Y-X*X)**2
    Z = np.log10(np.maximum(F, 1e-20))
    contour = base('', 'x', 'y')
    contour.add_trace(go.Heatmap(x=x, y=y, z=Z,
        colorscale=[[0, '#163c49'], [.4, '#17364b'], [1, '#101b2b']], zmin=-4, zmax=5,
        colorbar=dict(title='log₁₀ f'), hovertemplate='x=%{x:.3f}<br>y=%{y:.3f}<br>log₁₀ f=%{z:.2f}<extra></extra>'))
    # Exact level curves avoid false loops from interpolating a sampled valley.
    # f=c implies y=x² ± sqrt(c-(1-x)²)/10.
    upper_exponent = np.ceil(np.log10(max(float(F.max()), 1e-20))*2)/2
    lower_exponent = max(-20., min(-4., upper_exponent-8))
    for exponent in np.arange(lower_exponent, upper_exponent+.25, .5):
        level = 10.**exponent
        root = np.sqrt(level)
        left, right = max(low[0], 1-root), min(high[0], 1+root)
        if left >= right:
            continue
        # Cosine spacing resolves the tips where the two branches meet.
        u = left+(right-left)*(1-np.cos(np.linspace(0, np.pi, 600)))/2
        offset = np.sqrt(np.maximum(level-(1-u)**2, 0))/10
        xx = np.concatenate([u, [np.nan], u[::-1]])
        yy = np.concatenate([u*u+offset, [np.nan], (u*u-offset)[::-1]])
        contour.add_trace(go.Scatter(x=xx, y=yy, mode='lines', showlegend=False,
            hoverinfo='skip', line=dict(color='#3c5c71', width=.7, simplify=False)))
    values, gradients = base('Objective value', y='f(x, y)'), base('Gradient norm', y='‖∇f‖₂')
    rows = []
    for r in runs:
        history = r['history']
        color = COLORS[METHODS.index(r['method'])]
        visible = history[:iteration+1]
        contour.add_trace(go.Scatter(x=[p['x'][0] for p in visible], y=[p['x'][1] for p in visible],
            name=r['method'], mode='lines+markers', marker=dict(color=color, size=4), line=dict(color=color, width=2)))
        p = visible[-1]
        if 'simplex' in p:
            triangle = p['simplex']+[p['simplex'][0]]
            contour.add_trace(go.Scatter(x=[v[0] for v in triangle], y=[v[1] for v in triangle],
                mode='lines+markers', name='Nelder–Mead simplex', showlegend=False,
                fill='toself', fillcolor='rgba(242,127,167,0.10)', line=dict(color=color, width=1),
                marker=dict(size=5, symbol='diamond'),
                hovertemplate=p['operation']+'<br>x=%{x:.6f}<br>y=%{y:.6f}<extra></extra>'))
        contour.add_trace(go.Scatter(x=[p['x'][0]], y=[p['x'][1]], mode='markers', showlegend=False,
            marker=dict(color=color, size=10, line=dict(color='white', width=1)),
            hovertemplate=f"{r['method']}<br>iteration {len(visible)-1}<br>f={p['f']:.4g}<extra></extra>"))
        for fig, key in [(values, 'f'), (gradients, 'norm')]:
            fig.add_trace(go.Scatter(x=list(range(len(visible))), y=[max(item[key], 1e-20) for item in visible],
                                    name=r['method'], mode='lines+markers', marker=dict(size=4), line=dict(color=color)))
            fig.add_trace(go.Scatter(x=[len(visible)-1], y=[max(p[key], 1e-20)],
                                    mode='markers', showlegend=False, name=r['method'],
                                    marker=dict(color=color, size=10, line=dict(color='white', width=1))))
        last = history[-1]
        rows.append(html.Tr([html.Td(r['method'], style={'color': color}), html.Td(r['status']),
            html.Td(len(history)-1), html.Td(f"{last['f']:.3e}"), html.Td(f"{last['norm']:.3e}"),
            html.Td(f"({last['x'][0]:.6f}, {last['x'][1]:.6f})"),
            html.Td('/'.join(str(r['counts'][k]) for k in ('function', 'gradient', 'hessian')))]))
        rows[-1].children.extend([html.Td(r.get('rejected', '—')), html.Td(r.get('line_search', '—'))])
    if runs:
        start = runs[0].get('start', runs[0]['history'][0]['x'])
        contour.add_trace(go.Scatter(x=[start[0]], y=[start[1]], name='Start', mode='markers', marker=dict(symbol='diamond-open', size=14, color='white', line=dict(width=2))))
    contour.add_trace(go.Scatter(x=[1], y=[1], name='Optimum (1, 1)', mode='markers', marker=dict(symbol='star', size=17, color='#f8e6a0')))
    contour.update_layout(height=560, uirevision=str(runs[0]['history'][0]['x']) if runs else '', clickmode='event')
    contour.update_xaxes(range=[low[0], high[0]], autorange=False)
    contour.update_yaxes(range=[low[1], high[1]], autorange=False)
    # Reapply the browser's latest viewport: each frame creates a fresh figure.
    # Plotly emits split range keys for box zoom/pan, or autorange on reset.
    for axis in ('xaxis', 'yaxis'):
        view = viewport or {}
        if view.get(f'{axis}.autorange') is True:
            index = 0 if axis == 'xaxis' else 1
            contour.update_layout(**{axis: dict(range=[low[index], high[index]], autorange=False)})
        else:
            bounds = view.get(f'{axis}.range')
            if bounds is None and all(f'{axis}.range[{i}]' in view for i in (0, 1)):
                bounds = [view[f'{axis}.range[0]'], view[f'{axis}.range[1]']]
            if bounds is not None:
                contour.update_layout(**{axis: dict(range=bounds, autorange=False)})
    for fig in (values, gradients):
        fig.update_yaxes(type='log')
        fig.update_layout(height=300, showlegend=False)
    maximum = max((len(r['history'])-1 for r in runs), default=1)
    for fig, key in [(values, 'f'), (gradients, 'norm')]:
        all_values = [max(p[key], 1e-20) for r in runs for p in r['history']]
        if all_values:
            lo, hi = np.log10(min(all_values)), np.log10(max(all_values))
            margin = max(.3, (hi-lo)*.06)
            fig.update_yaxes(range=[lo-margin, hi+margin])
        fig.update_xaxes(range=[0, max(1, maximum)])
    table = html.Table([html.Thead(html.Tr([html.Th(s) for s in ['Method', 'Status', 'Steps', 'Final f', 'Final ‖∇f‖', 'Endpoint', 'f / g / H evals', 'TR rejected', 'Line search']])), html.Tbody(rows)])
    return contour, values, gradients, table, f'Playback iteration {iteration:,}. All plots reveal accepted steps together; the table shows final results.'


@app.callback(Output('download', 'data'), Input('export', 'n_clicks'), State('runs', 'data'), prevent_initial_call=True)
def export(_, runs):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['method', 'iteration', 'x', 'y', 'objective', 'gradient_norm', 'step_length', 'trust_radius', 'reduction_ratio', 'simplex_diameter', 'objective_spread', 'simplex_operation'])
    for r in runs or []:
        for k, p in enumerate(r['history']):
            writer.writerow([r['method'], k, *p['x'], p['f'], p['norm'], p['alpha'], p.get('radius'), p.get('rho'), p.get('diameter'), p.get('spread'), p.get('operation')])
    return dict(content=output.getvalue(), filename='rosenbrock-trajectories.csv')


if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=8050)
