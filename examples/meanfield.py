"""Native Torch VI: a diagonal guide, exact references, and unknown noise.

Run: python -m examples.meanfield
Requires torch and viz extras, but does not import Pyro.
"""
from pathlib import Path
import json
import math
import numpy as np
from smallprob import model, Normal, HalfNormal
from examples.vi_validation import gaussian_regression_reference, quadrature_regression_reference


@model
def regression(x):
    b = Normal(0, 5)
    m = Normal(0, 5)
    sigma = HalfNormal(1)
    y = Normal(b + m * x, sigma)


@model
def known_noise(x):
    b = Normal(0, 5)
    m = Normal(0, 5)
    y = Normal(b + m * x, .6)


def run():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from smallprob.plotting import use_style
    use_style()
    root = Path(__file__).resolve().parents[1]
    target = root / 'website/assets/meanfield'
    target.mkdir(parents=True, exist_ok=True)

    x = np.linspace(-1.5, 1.5, 32)
    bound = regression(x)
    data = bound.simulate(seed=7, fixed={'b': 1.2, 'm': -.7, 'sigma': .6})
    y = data['y'][0]
    posterior = bound.infer(y=y, method='vi', guide='diagonal', steps=4000, seed=7)
    reference = quadrature_regression_reference(np.column_stack([np.ones_like(x), x]), y)
    refined = quadrature_regression_reference(np.column_stack([np.ones_like(x), x]), y, bounds=(-10, 4), points=8193)
    np.testing.assert_allclose(reference['mean'], refined['mean'], atol=2e-6)
    names = ['b', 'm', 'sigma']
    unknown = dict(mean=[float(posterior.mean(n)) for n in names], sd=[float(posterior.sd(n)) for n in names],
                   reference_mean=reference['mean'].tolist(), reference_sd=reference['sd'].tolist())
    unknown['max_mean_error_reference_sd'] = float(np.max(np.abs(np.array(unknown['mean'])-reference['mean'])/reference['sd']))
    unknown['max_sd_relative_error'] = float(np.max(np.abs(np.array(unknown['sd'])/reference['sd']-1)))
    if unknown['max_mean_error_reference_sd'] >= .35:
        raise AssertionError(f"Unknown-noise mean check failed: {unknown}")

    runs = []
    for centered in [True, False]:
        values = x if centered else x + 1.5
        design = np.column_stack([np.ones_like(x), values])
        model_ = known_noise(values)
        ref = gaussian_regression_reference(design, y, .6)
        precision = design.T @ design / .6**2 + np.eye(2)/25
        optimal_sd = 1/np.sqrt(np.diag(precision))
        for seed in [0, 7, 19]:
            fit = model_.infer(y=y, method='vi', guide='diagonal', steps=3000, seed=seed)
            means = np.array([fit.mean(n) for n in ['b', 'm']])
            sds = np.array([fit.sd(n) for n in ['b', 'm']])
            entry = dict(centered=centered, seed=seed, mean=means.tolist(), sd=sds.tolist(),
                         exact_mean=ref['mean'].tolist(), exact_sd=ref['sd'].tolist(),
                         diagonal_optimum_sd=optimal_sd.tolist(),
                         max_mean_error_reference_sd=float(np.max(np.abs(means-ref['mean'])/ref['sd'])),
                         max_sd_error_diagonal_optimum=float(np.max(np.abs(sds/optimal_sd-1))))
            assert entry['max_mean_error_reference_sd'] < .2, entry
            assert entry['max_sd_error_diagonal_optimum'] < .15, entry
            runs.append(entry)
        if not centered:
            correlated_ref, correlated_fit = ref, fit

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    def ellipse(mean, cov, color, label):
        eigenvalues, vectors = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
        scale = math.sqrt(-2*math.log(.1))  # 90% joint Gaussian ellipse (df=2)
        axes[0].add_patch(Ellipse(mean, 2*scale*np.sqrt(eigenvalues[1]), 2*scale*np.sqrt(eigenvalues[0]), angle=angle,
                                 facecolor=color, alpha=.14, edgecolor=color, lw=2, label=label))
        axes[0].plot(*mean, '.', color=color)
    ellipse(correlated_ref['mean'], correlated_ref['covariance'], '#246892', 'Exact joint posterior: 90%')
    ellipse(np.array([correlated_fit.mean(n) for n in ['b', 'm']]), np.diag([correlated_fit.sd(n)**2 for n in ['b', 'm']]), '#b52d24', 'Diagonal guide: 90%')
    sd = correlated_ref['sd']
    axes[0].set(xlim=correlated_ref['mean'][0]+np.array([-2.5, 2.5])*sd[0], ylim=correlated_ref['mean'][1]+np.array([-2.5, 2.5])*sd[1],
                xlabel='Intercept b', ylabel='Slope m', title='When posterior correlation matters')
    axes[0].legend(loc='upper right', fontsize=8)
    draws = posterior.sample(4000, seed=8)
    lines = draws['b'][:, None] + draws['m'][:, None]*x
    lo, hi = np.quantile(lines, [.025, .975], axis=0)
    axes[1].scatter(x, y, s=18, color='#525650', alpha=.65, label='Synthetic observations')
    axes[1].plot(x, lines.mean(0), color='#b52d24', label='Mean-field mean response')
    axes[1].fill_between(x, lo, hi, color='#b52d24', alpha=.15, label='95% guide band for mean')
    axes[1].set(xlabel='x', ylabel='y', title='Learn coefficients and noise scale')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(target/'validation.svg', metadata={'Date': None})
    fig.savefig(target/'validation.png', dpi=300)
    plt.close(fig)
    report = dict(backend='native PyTorch; Pyro not imported', guide='independent Normal factors; lognormal positive sites',
                  seeds=[0, 7, 19], known_noise=runs, unknown_noise=unknown,
                  caveat='The mean-field optimum has variance 1 / diag(posterior precision), not diag(posterior covariance).',
                  unknown_reference='Independent numerical quadrature, refined from 4097 to 8193 points and wider bounds.',
                  thresholds={'known_mean_reference_sd':.2, 'sd_diagonal_optimum_relative':.15, 'unknown_mean_reference_sd':.35},
                  passed=True)
    (target/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'known_noise_runs':len(runs), 'max_mean_error_reference_sd':max(r['max_mean_error_reference_sd'] for r in runs),
                      'max_sd_error_diagonal_optimum':max(r['max_sd_error_diagonal_optimum'] for r in runs), 'unknown_noise':unknown, 'passed':True}, indent=2))
    return report


if __name__ == '__main__':
    run()
