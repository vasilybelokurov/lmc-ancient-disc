"""Forward response for the metallicity-to-mass sensitivity diagnostic.

Columns are uniform true-metallicity bins; rows are measured-metallicity bins.
Mass is formed stellar mass. Selection-corrected counts have expectation
R @ (eta * mass), where eta is observable giants per unit formed mass.
`response` holds assumed kernels for the sensitivity diagnostic; `empirical_response` is measured by the caller from stars with
a reference (true) and a measured [Fe/H].

The response matrix R has shape (n_measured_bins, n_true_bins); column j is the distribution over measured bins of stars whose
true [Fe/H] is in bin j, so each column sums to 1 when the measured edges cover all values. recover_mass inverts
counts ~ R @ (eta * mass). Used by scripts/build_giant_mass.py (BOSS masses per true [Fe/H] bin), scripts/boss_response.py
(figures/paper_boss_response.png), lmcold.mass_samples, and the mock diagnostics (scripts/fig_mass_recovery.py, fig_age_mock.py).
"""
import numpy as np
from scipy.optimize import minimize, nnls
from scipy.special import ndtr, xlogy


def response(true_edges, measured_edges, fractions, sigmas, tail_fraction=0., tail_shift=-.8):
    """Gaussian errors in magnitude groups; optional faint metal-rich outliers.

Last magnitude group is faint. Its true [Fe/H] > -1.5 stars have probability
`tail_fraction` of an additional `tail_shift` dex. Integrate uniformly within
true bins with 200 midpoint samples. Include infinite measured end bins.

Parameters
----------
true_edges, measured_edges : array_like
    Bin edges [dex]; measured_edges should start at -inf and end at +inf for the columns to sum to 1.
fractions : array_like
    Fraction of stars in each magnitude group (non-negative, summing to 1).
sigmas : array_like
    Gaussian [Fe/H] error of each group [dex]; 0 gives a step (no error).
tail_fraction : float
    Probability of the shifted (outlier) component for metal-rich stars of the last group.
tail_shift : float
    Shift of that component [dex].

Returns
-------
ndarray, shape (len(measured_edges) - 1, len(true_edges) - 1)
"""
    fractions, sigmas = np.asarray(fractions), np.asarray(sigmas)
    if not np.isclose(fractions.sum(), 1) or np.any(fractions < 0):
        raise ValueError('Magnitude fractions must be nonnegative and sum to one')
    out = np.zeros((len(measured_edges) - 1, len(true_edges) - 1))
    for j, (lo, hi) in enumerate(zip(true_edges[:-1], true_edges[1:])):
        f = lo + (hi - lo) * (np.arange(200) + .5) / 200
        for k, (p, sd) in enumerate(zip(fractions, sigmas)):
            def kernel(shift):
                """Probability per measured bin (rows) for each of the 200 true values (columns), errors shifted by shift."""
                if sd == 0:
                    return np.diff((np.asarray(measured_edges)[:, None] > f + shift).astype(float), axis=0)
                return np.diff(ndtr((np.asarray(measured_edges)[:, None] - f - shift) / sd), axis=0)
            fraction = tail_fraction * (f > -1.5) if k == len(fractions) - 1 else 0.
            out[:, j] += p * np.mean((1 - fraction) * kernel(0) + fraction * kernel(tail_shift), axis=1)
    return out


def empirical_response(true_feh, measured_feh, weight, true_edges, measured_edges, nmin=15):
    """Response measured on calibration stars: column j is the weighted histogram of measured [Fe/H] of the stars whose true
    [Fe/H] is in true bin j. The measured values keep any dependence of the errors on the true value (e.g. shrinkage towards
    the training prior of a neural-network pipeline), which a model of independent errors would miss. A column with fewer than
    nmin stars borrows the stars of the nearest bins, each moved by the difference of the bin centres.

    Parameters
    ----------
    true_feh, measured_feh : array_like, shape (N,)
        Reference and measured [Fe/H] [dex] of the calibration stars (true values outside the edges go to the end bins).
    weight : array_like, shape (N,)
        Weight of each star.
    true_edges, measured_edges : array_like
        Bin edges [dex].
    nmin : int
        Smallest number of stars per column (the window widens by one bin on each side until reached; the total number of
        stars must be at least nmin).

    Returns
    -------
    ndarray, shape (len(measured_edges) - 1, len(true_edges) - 1)
        Each column normalised to 1 over the measured bins.
    """
    edges = np.asarray(true_edges)
    centre = .5 * (edges[1:] + edges[:-1])
    k = np.digitize(np.clip(true_feh, edges[0], edges[-1] - 1e-9), edges) - 1
    out = np.zeros((len(measured_edges) - 1, len(centre)))
    for j in range(len(centre)):
        n = 0
        m = k == j
        while m.sum() < nmin:
            n += 1
            m = np.abs(k - j) <= n
        # Borrowed stars are shifted by the distance between their bin centre and bin j.
        h = np.histogram(np.asarray(measured_feh)[m] + centre[j] - centre[k[m]], measured_edges, weights=np.asarray(weight)[m])[0]
        out[:, j] = h / h.sum()
    return out


def recover_mass(counts, response_matrix, eta):
    """Nonnegative Poisson mean fit to selection-weighted counts.

    This is a quasi-likelihood: weighted counts are not independent Poisson
    draws. Its mean equation avoids observed-variance least-squares bias;
    uncertainty here comes only from repeated full mock realisations.
    No regularisation or claim of a production estimator.

    Minimises the Poisson deviance sum(mu - n + n ln(n / mu)) with mu = R @ (eta * mass), mass >= 0 (L-BFGS-B, started from
    the non-negative least-squares solution).

    Parameters
    ----------
    counts : ndarray, shape (n_measured,)
        Selection-corrected giant counts per measured [Fe/H] bin.
    response_matrix : ndarray, shape (n_measured, n_true)
        R (response or empirical_response).
    eta : array_like, shape (n_true,)
        Giants per Msun formed in each true bin (lmcold.mass_conversion.factors).

    Returns
    -------
    ndarray, shape (n_true,)
        Formed mass per true [Fe/H] bin [Msun].

    Raises
    ------
    RuntimeError
        If the optimiser does not converge.
    """
    # Work in units of 1e6 Msun so the fitted masses are of order unity.
    design = response_matrix * np.asarray(eta)[None, :] * 1e6
    initial, _ = nnls(design, counts, maxiter=10000)
    norm = max(np.sum(counts), 1.)  # the loss is per unit total count

    def objective(mass):
        """Deviance divided by the total count (norm) and its gradient with respect to mass [1e6 Msun]."""
        expected = design @ mass + 1e-12  # expected counts mu; 1e-12 keeps log and division finite at mu = 0
        loss = np.sum(expected - counts + xlogy(counts, counts / expected)) / norm
        gradient = design.T @ (1 - counts / expected) / norm
        return loss, gradient

    fit = minimize(objective, initial, jac=True, method='L-BFGS-B',
                   bounds=[(0, None)] * len(eta),
                   options=dict(ftol=1e-14, gtol=1e-10, maxiter=10000, maxls=50))
    if not fit.success:
        raise RuntimeError(f'Mass response fit failed: {fit.message}')
    return fit.x * 1e6
