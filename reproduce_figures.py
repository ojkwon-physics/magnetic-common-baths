#!/usr/bin/env python3
"""
Reproduce figures for:
  O. J. Kwon, "Finite-frequency magnetic common baths in ferromagnetic planar cavities"

Usage
-----
  python reproduce_figures.py               # load pre-computed CSVs, generate figures
  python reproduce_figures.py --recompute   # recompute from physics, update CSVs, generate figures
  python reproduce_figures.py --outdir DIR  # write PDF/PNG figures to DIR (default: figures/)
  python reproduce_figures.py --validate    # compare CSVs with recomputation and check static limits
  python reproduce_figures.py --report      # print the cavity-induced linewidth ratio
  python reproduce_figures.py --validate --validation-json validation_results.json
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.integrate import trapezoid
from scipy.special import j0, jv

# ---------------------------------------------------------------------------
# Physical constants and benchmark parameters
# ---------------------------------------------------------------------------
mu0    = 4.0 * np.pi * 1e-7
eps0   = 8.8541878128e-12
hbar   = 1.054571817e-34
kB     = 1.380649e-23
c      = 299792458.0
muB    = 9.2740100783e-24
ge     = 2.00231930436256

f_probe  = 2.87e9
omega    = 2.0 * np.pi * f_probe
k0       = omega / c
T_bench  = 0.020
nth      = 1.0 / (np.exp(hbar * omega / (kB * T_bench)) - 1.0)

L        = 3.0e-6
zA = zB  = L / 2.0
film_t   = 200e-9
sigma_Ni = 1.43e7
mu0_Ms   = 0.617
mu0_Meff = 0.617
alpha    = 0.064
gammaF   = 2.0 * np.pi * 28.0e9
m_c      = ge * muB / np.sqrt(2.0)

B_on       = 16.5e-3
BVALS_3    = [8.0e-3, B_on, 40.0e-3]
LABELS_3   = ["8 mT", "16.5 mT", "40 mT"]

# ---------------------------------------------------------------------------
# APS/PRA plotting style
# ---------------------------------------------------------------------------
COLORS = ["#1A5276", "#B03A2E", "#1E8449", "#6C3483", "#CA6F1E"]
REF_COLOR = "0.25"

mpl.rcParams.update({
    "axes.prop_cycle":    mpl.cycler(color=COLORS),
    "font.family":        "DejaVu Serif",
    "mathtext.fontset":   "dejavuserif",
    "font.size":          10.0,
    "axes.labelsize":     10.0,
    "axes.titlesize":     8.4,
    "xtick.labelsize":    9.0,
    "ytick.labelsize":    9.0,
    "legend.fontsize":    8.6,
    "axes.linewidth":     0.95,
    "lines.linewidth":    1.45,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.top":          True,
    "ytick.right":        True,
    "xtick.major.size":   3.2,
    "ytick.major.size":   3.2,
    "xtick.major.width":  0.85,
    "ytick.major.width":  0.85,
    "savefig.dpi":        600,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

FIGURE_LABELSIZE = 9.0
FIGURE_TICKSIZE = 8.5
FIGURE_LEGENDSIZE = 8.5

# ---------------------------------------------------------------------------
# Electromagnetic response functions
# ---------------------------------------------------------------------------
def eps_metal(sigma: float = sigma_Ni) -> complex:
    return 1.0 + 1j * sigma / (eps0 * omega)


def mu_perp(B: np.ndarray | float) -> np.ndarray:
    # Effective Polder-type scalar permeability, not a diagonal component of the
    # full in-plane gyrotropic susceptibility.
    B      = np.asarray(B, dtype=float)
    omega0 = gammaF * np.sqrt(B * (B + mu0_Meff))
    omegaM = gammaF * mu0_Ms
    damped = omega0 - 1j * alpha * omega
    return 1.0 + omegaM * damped / (damped * damped - omega * omega)


def kappa_vacuum(k: np.ndarray) -> np.ndarray:
    k  = np.asarray(k, dtype=float)
    ev = k >= k0
    out = np.empty_like(k, dtype=complex)
    out[ev]  = np.sqrt(k[ev]  * k[ev]  - k0 * k0)
    out[~ev] = -1j * np.sqrt(k0 * k0  - k[~ev] * k[~ev])
    return out


def kappa_layer(k: np.ndarray,
                eps: complex | np.ndarray,
                mu:  complex | np.ndarray) -> np.ndarray:
    z   = np.asarray(k, dtype=float) ** 2 - eps * mu * k0 * k0
    out = np.sqrt(z + 0j)
    flip = (out.real < 0.0) | ((np.abs(out.real) < 1.0e-30) & (out.imag > 0.0))
    return np.where(flip, -out, out)


def r_te_slab(k: np.ndarray, B: float | np.ndarray,
              thickness: float = film_t,
              half_space: bool = False) -> np.ndarray:
    """Reciprocal scalar TE reflection amplitude for vacuum | film | substrate."""
    k = np.asarray(k, dtype=float)
    B = np.asarray(B, dtype=float)
    if B.ndim > 0:
        kk, BB = k[:, None], B[None, :]
    else:
        kk, BB = k, float(B)
    eps1 = eps_metal();  mu1 = mu_perp(BB)
    eps2 = eps_metal();  mu2 = 1.0
    kap0 = kappa_vacuum(kk)
    kap1 = kappa_layer(kk, eps1, mu1)
    kap2 = kappa_layer(kk, eps2, mu2)
    r01  = (mu1 * kap0 - kap1) / (mu1 * kap0 + kap1)
    if half_space:
        return r01
    r12   = (mu2 * kap1 - mu1 * kap2) / (mu2 * kap1 + mu1 * kap2)
    phase = np.exp(-2.0 * kap1 * thickness)
    return (r01 + r12 * phase) / (1.0 + r01 * r12 * phase)


def green_scalar(rho: np.ndarray | float, B: float,
                 projection: str = "zz",
                 qmin: float = 1.0e-4, qmax: float = 80.0,
                 nq: int = 6000) -> np.ndarray:
    """Scattering magnetic Green-tensor projection in the scalar TE channel.

    Grid: q = k_perp * L uniform on [qmin, qmax] with nq points.
    The figure defaults reproduce Appendix D; qmin is exposed so that the
    propagating sector q < k0*L can be integrated separately as a diagnostic.
    projection='zz'       -> G_BB,zz
    projection='xx'       -> G_BB,xx (TE contribution)
    projection='yy'       -> G_BB,yy (TE contribution)
    projection='perp_avg' -> (G_xx + G_yy) / 2 (TE contribution)
    """
    scalar_input = np.ndim(rho) == 0
    rho   = np.atleast_1d(np.asarray(rho, dtype=float))
    q     = np.linspace(qmin, qmax, nq)
    k     = q / L
    kap0  = kappa_vacuum(k)
    rs    = r_te_slab(k, B)
    Sigma = zA + zB
    Dz    = abs(zA - zB)
    odd   = rs * (np.exp(-kap0 * Sigma) + np.exp(-kap0 * (2.0 * L - Sigma)))
    even  = rs * rs * (np.exp(-kap0 * (2.0 * L - Dz)) + np.exp(-kap0 * (2.0 * L + Dz)))
    denom = 1.0 - rs * rs * np.exp(-2.0 * kap0 * L)
    x = np.outer(k, rho)
    if projection == "zz":
        angular_kernel = j0(x)
        curl_prefactor = k * k / (2.0 * kap0)
        numerator = odd + even
    elif projection in {"xx", "yy", "perp_avg"}:
        # The in-plane component of the TE magnetic polarization reverses between
        # upward and downward branches, while the normal component does not.
        # Odd- and even-reflection paths therefore enter the transverse average
        # with opposite relative signs.  Complete round trips add two
        # reflections and preserve parity, so the denominator is unchanged.
        if projection == "xx":
            angular_kernel = j0(x) - jv(2, x)
        elif projection == "yy":
            angular_kernel = j0(x) + jv(2, x)
        else:
            angular_kernel = j0(x)
        curl_prefactor = kap0 / 4.0
        numerator = odd - even
    else:
        raise ValueError("projection must be 'zz', 'xx', 'yy', or 'perp_avg'")
    pref = (k / (2.0 * np.pi)) * curl_prefactor * numerator / denom
    values = trapezoid(pref[:, None] * angular_kernel, k, axis=0)
    return values[0] if scalar_input else values


def gamma_mHz(G: np.ndarray) -> np.ndarray:
    return (2.0 * mu0 * m_c * m_c / hbar) * np.imag(G) / (2.0 * np.pi) * 1.0e3


def omega_mHz(G: np.ndarray) -> np.ndarray:
    # The minus sign follows the Hamiltonian convention
    # H_LS = +hbar sum_ij Omega_ij sigma_i^+ sigma_j^-, which fixes the
    # exchange kernel as -(mu0/hbar) m* . Re G_BB . m.
    return -(mu0 * m_c * m_c / hbar) * np.real(G) / (2.0 * np.pi) * 1.0e3


def static_tensor_kernel(a: float, rho: float, projection: str) -> float:
    """Real-space static tensor kernel for one image height."""
    r2 = a * a + rho * rho
    if projection == "zz":
        return (2.0 * a * a - rho * rho) / r2 ** 2.5
    if projection == "xx":
        return (2.0 * rho * rho - a * a) / r2 ** 2.5
    if projection == "yy":
        return -1.0 / r2 ** 1.5
    if projection == "perp_avg":
        return 0.5 * (
            (2.0 * rho * rho - a * a) / r2 ** 2.5 - 1.0 / r2 ** 1.5
        )
    raise ValueError("projection must be 'zz', 'xx', 'yy', or 'perp_avg'")


def static_integral_constant_r(rho: np.ndarray, r: float = 0.65,
                               projection: str = "zz",
                               qmax: float = 200.0,
                               nq: int = 50000) -> np.ndarray:
    """Constant-r static k-space kernel for a selected tensor projection."""
    rho = np.atleast_1d(rho)
    q = np.linspace(1.0e-6, qmax, nq)
    odd = 2.0 * r * np.exp(-q)
    even = 2.0 * r * r * np.exp(-2.0 * q)
    den = 1.0 - r * r * np.exp(-2.0 * q)
    if projection == "zz":
        numerator = odd + even
        prefactor = q * q / 2.0
    elif projection in {"xx", "yy", "perp_avg"}:
        numerator = odd - even
        prefactor = q * q / 4.0
    else:
        raise ValueError("projection must be 'zz', 'xx', 'yy', or 'perp_avg'")

    values = []
    for rr in rho:
        x = q * rr / L
        if projection == "xx":
            angular_kernel = j0(x) - jv(2, x)
        elif projection == "yy":
            angular_kernel = j0(x) + jv(2, x)
        else:
            angular_kernel = j0(x)
        values.append(
            (1.0 / (2.0 * np.pi * L ** 3))
            * trapezoid(prefactor * angular_kernel * numerator / den, q)
        )
    return np.asarray(values)


def static_image_constant_r(rho: np.ndarray, r: float = 0.65,
                            projection: str = "zz",
                            nmax: int = 360) -> np.ndarray:
    """Constant-r real-space image series with reflection parity included."""
    rho = np.atleast_1d(rho)
    dz = abs(zA - zB)
    sz = zA + zB
    odd_orientation = 1.0 if projection == "zz" else -1.0
    out = np.zeros_like(rho, dtype=float)
    for i, rr in enumerate(rho):
        value = 0.0
        for n in range(nmax):
            value += odd_orientation * r ** (2 * n + 1) * (
                static_tensor_kernel(sz + 2 * n * L, rr, projection)
                + static_tensor_kernel(2 * L - sz + 2 * n * L, rr, projection)
            )
        for n in range(1, nmax):
            value += r ** (2 * n) * (
                static_tensor_kernel(2 * n * L - dz, rr, projection)
                + static_tensor_kernel(2 * n * L + dz, rr, projection)
            )
        out[i] = value / (4.0 * np.pi)
    return out

# ---------------------------------------------------------------------------
# Data computation
# ---------------------------------------------------------------------------
def compute_kernel_data(projection: str) -> np.ndarray:
    """Compute kernel curves over rho for 3 bias values.

    Returns (220, 7): rho_um | gam_8 gam_165 gam_40 | om_8 om_165 om_40
    Units: gamma and omega in mHz.
    """
    rho_um = np.linspace(1.2, 8.0, 220)
    rho    = rho_um * 1e-6
    gams, oms = [], []
    for B in BVALS_3:
        G = green_scalar(rho, B, projection=projection)
        gams.append(gamma_mHz(G))
        oms.append(omega_mHz(G))
    return np.column_stack([rho_um] + gams + oms)


def compute_data_fig4() -> np.ndarray:
    """Compute bias-scan kernel at rho = 3 um.

    Returns (300, 3): B_mT | gam_zz_mHz | om_zz_mHz
    """
    B_mT = np.linspace(3.0, 70.0, 300)
    rho  = 3.0e-6
    G    = np.array([green_scalar(rho, b * 1e-3, projection="zz") for b in B_mT])
    return np.column_stack([B_mT, gamma_mHz(G), omega_mHz(G)])


def compute_data_figA1() -> np.ndarray:
    """Compute static-limit relative difference.

    Returns (150, 2): rho_um | rel_diff
    """
    rho_um = np.linspace(1.0, 10.0, 150)
    rho    = rho_um * 1e-6
    gk     = static_integral_constant_r(rho, projection="zz")
    gi     = static_image_constant_r(rho, projection="zz")
    return np.column_stack([rho_um, np.abs((gk - gi) / gi)])


def static_projection_check(projection: str) -> float:
    """Maximum relative k-space/image-series discrepancy on the plotted grid."""
    rho = np.linspace(1.0e-6, 10.0e-6, 150)
    gk = static_integral_constant_r(rho, projection=projection)
    gi = static_image_constant_r(rho, projection=projection)
    return float(np.max(np.abs((gk - gi) / gi)))

# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------
HDR_KERNEL = "rho_um,gam_8mT,gam_165mT,gam_40mT,om_8mT,om_165mT,om_40mT"
HDR_FIG4   = "B_mT,gam_zz_mHz,om_zz_mHz"
HDR_FIGA1  = "rho_um,rel_diff"


def save_csv(arr: np.ndarray, path: Path, header: str) -> None:
    np.savetxt(path, arr, delimiter=",", header=header, comments="")
    print(f"  saved {path.name}")


def load_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1)

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def panel_label(ax, label, fontsize=10.0):
    ax.text(0.0, 1.04, label, transform=ax.transAxes,
            ha="left", va="bottom", fontweight="bold", fontsize=fontsize)


def style_axes(axes):
    for ax in np.ravel(axes):
        ax.xaxis.label.set_size(FIGURE_LABELSIZE)
        ax.yaxis.label.set_size(FIGURE_LABELSIZE)
        ax.tick_params(axis="both", labelsize=FIGURE_TICKSIZE)
        ax.grid(False)


def signed_scale(ax, threshold, ticks):
    ax.set_yscale("symlog", linthresh=threshold)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.axhline(0, color="0.5", lw=0.7)


def free_exchange_mHz(rho):
    """Free-space zz exchange for equal-height, in-plane-separated moments."""
    rho = np.asarray(rho, dtype=float)
    x = k0 * rho
    G = np.exp(1j*x) * (x*x + 1j*x - 1) / (4*np.pi*rho**3)
    return omega_mHz(G)


def savefig(fig: mpl.figure.Figure, outdir: Path,
            name: str, write_png: bool = True) -> None:
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
    if write_png:
        fig.savefig(outdir / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------
def make_fig2(outdir: Path) -> None:
    """Generate finite-slab and half-space TE reflection maps."""
    Bvals = np.linspace(3e-3, 70e-3, 120)
    qvals = np.geomspace(1e-2, 30.0, 100)
    kk = qvals / L
    R_slab = np.empty((qvals.size, Bvals.size))
    R_inf = np.empty_like(R_slab)
    for j, B in enumerate(Bvals):
        R_slab[:, j] = np.abs(r_te_slab(kk, B, half_space=False))
        R_inf[:, j] = np.abs(r_te_slab(kk, B, half_space=True))

    log_slab = np.log10(np.maximum(R_slab, 1e-9))
    log_inf = np.log10(np.maximum(R_inf, 1e-9))
    vmin = min(np.nanmin(log_slab), np.nanmin(log_inf))
    vmax = max(np.nanmax(log_slab), np.nanmax(log_inf))

    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.7),
                           width_ratios=(1.0, 1.0, 1.05), layout="constrained")

    im0 = ax[0].pcolormesh(Bvals * 1e3, qvals, log_slab, shading="gouraud", vmin=vmin, vmax=vmax)
    ax[1].pcolormesh(Bvals * 1e3, qvals, log_inf, shading="gouraud", vmin=vmin, vmax=vmax)

    for a in ax[:2]:
        a.set_yscale("log")
        a.set_xlabel(r"$B$ (mT)")
        a.set_ylabel(r"$k_\perp L$")
        a.tick_params(axis="both", which="major", labelsize=8.8)

    for a in ax[:2]:
        a.axvline(B_on * 1e3, color="white", ls=":", lw=1.0)
    panel_label(ax[0], "(a)", fontsize=9.8)
    panel_label(ax[1], "(b)", fontsize=9.8)

    text_bbox = dict(facecolor="white", alpha=0.68, edgecolor="none", pad=1.3)
    ax[0].text(0.95, 0.055, r"finite film", transform=ax[0].transAxes,
               ha="right", va="bottom", fontsize=8.4, bbox=text_bbox)
    ax[1].text(0.95, 0.055, r"$t\to\infty$", transform=ax[1].transAxes,
               ha="right", va="bottom", fontsize=8.4, bbox=text_bbox)

    cb = fig.colorbar(im0, ax=[ax[0], ax[1]], shrink=0.91, pad=0.018)
    cb.set_label(r"$\log_{10}|r_s|$", fontsize=9.0)
    cb.ax.tick_params(labelsize=8.3)

    finite_cut = np.abs(r_te_slab(kk, B_on, half_space=False))
    half_cut = np.abs(r_te_slab(kk, B_on, half_space=True))

    ax[2].plot(qvals, finite_cut, lw=1.45, label=r"finite film")
    ax[2].plot(qvals, half_cut, ls="--", lw=1.45, label=r"$t\to\infty$")
    ax[2].set_xscale("log")

    yline = np.r_[finite_cut, half_cut]
    ax[2].set_ylim(max(0.0, 0.983 * np.nanmin(yline)), 1.055 * np.nanmax(yline))
    ax[2].set_xlabel(r"$k_\perp L$")
    ax[2].set_ylabel(r"$|r_s|$")
    ax[2].tick_params(axis="both", which="major", labelsize=8.8)

    ax[2].set_title(r"$B=16.5$ mT", fontsize=8.5, loc="right", pad=9)
    panel_label(ax[2], "(c)", fontsize=9.8)

    ax[2].legend(frameon=True, loc="upper left", bbox_to_anchor=(0.06, 0.84),
                 fontsize=8.0, handlelength=1.45, labelspacing=0.25,
                 handletextpad=0.45, borderpad=0.25, borderaxespad=0.0,
                 framealpha=0.72, edgecolor="none", facecolor="white")

    savefig(fig, outdir, "Fig2_reflection_map")


def make_kernel_pair(data, outdir, axis, filename, note=None):
    """Plot cavity corrections; add the total exchange for the main zz result."""
    rho = data[:, 0]
    main = axis == "zz"
    if main:
        fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.75), layout="constrained")
    else:
        fig, ax = plt.subplots(2, 1, figsize=(3.4, 4.0), sharex=True, layout="constrained")
    markers = ["o", "s", "^"]
    for j, label in enumerate(LABELS_3):
        kw = dict(color=COLORS[j], lw=1.3, marker=markers[j],
                  markevery=35, ms=3, mfc="white", label=label)
        ax[0].plot(rho, data[:, j+1], **kw)
        ax[1].plot(rho, data[:, j+4], **kw)
        if main:
            ax[2].plot(rho, free_exchange_mHz(rho*1e-6)+data[:, j+4], **kw)
    threshold = max(np.max(abs(data[:, 1:4]))*.1, .01)
    signed_scale(ax[0], threshold, [-1, -.1, 0, .1, 1] if main else [-.1, 0, .1, 1])
    ax[1].axhline(0, color="0.5", lw=.7)
    proj = "zz" if main else r"\perp"
    ax[0].set_ylabel(rf"$\delta\gamma_{{12}}^{{({proj})}}/(2\pi)$ (mHz)")
    ax[1].set_ylabel(rf"$\Omega_{{12}}^{{\mathrm{{sc}},{proj}}}/(2\pi)$ (mHz)")
    titles = ["Cross-decay correction", "Exchange correction"]
    if main:
        ax[2].plot(rho, free_exchange_mHz(rho*1e-6), color="0.3", ls="--", lw=1.2, label="Free space")
        ax[2].set_yscale("log")
        ax[2].set_ylabel(r"$\Omega_{12}^{zz}/(2\pi)$ (mHz)")
        ax[2].legend(fontsize=7.8, frameon=False, loc="upper right", handlelength=1.6)
        titles.append("Total exchange")
    else:
        ax[0].legend(fontsize=FIGURE_LEGENDSIZE, frameon=False, loc="upper right")
    for j, a in enumerate(ax):
        panel_label(a, f"({chr(97+j)})")
        a.set_title(titles[j], fontsize=8.5, loc="right", pad=10)
        a.set_xlabel(r"$\rho$ ($\mu$m)")
    style_axes(ax)
    savefig(fig, outdir, filename)


def make_fig4(data, outdir):
    """Cavity corrections for ideal uniform ensemble modes, in hertz."""
    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.8), layout="constrained")
    for j, N in enumerate((1000, 5000)):
        kw = dict(color=COLORS[j], lw=1.4, marker=["o", "s"][j],
                  markevery=45, ms=3.5, mfc="white", label=rf"$N_{{\mathrm{{eff}}}}={N}$")
        ax[0].plot(data[:, 0], 2*N*data[:, 1]/1e3, **kw)
        ax[1].plot(data[:, 0], N*data[:, 2]/1e3, **kw)
    signed_scale(ax[0], max(np.max(abs(10*data[:, 1]))*.1, .01), [-10, -1, 0, 1, 10])
    ax[1].axhline(0, color="0.5", lw=.7)
    ax[0].set_ylabel(r"$2\delta\gamma_{AB}^{\mathrm{br},zz}/(2\pi)$ (Hz)")
    ax[1].set_ylabel(r"$\Omega_{AB}^{\mathrm{br,sc},zz}/(2\pi)$ (Hz)")
    for j, a in enumerate(ax):
        a.axvline(B_on*1e3, color="0.5", ls=":", lw=1)
        a.set_xlabel(r"$B$ (mT)")
        panel_label(a, f"({chr(97+j)})")
    ax[0].set_title("Linewidth-difference correction", loc="right", fontsize=9, pad=10)
    ax[1].set_title("Exchange correction", loc="right", fontsize=9, pad=10)
    ax[1].legend(frameon=False, fontsize=FIGURE_LEGENDSIZE)
    style_axes(ax)
    savefig(fig, outdir, "Fig4_bright_observables")


def make_figA1(data, outdir):
    """Show the static integral and image series, then their relative difference."""
    rho = data[:, 0]
    g_int = static_integral_constant_r(rho*1e-6)
    g_img = static_image_constant_r(rho*1e-6)
    # Multiplying by L^3 gives a dimensionless response in both representations.
    fig, ax = plt.subplots(2, 1, figsize=(3.4, 3.8), sharex=True,
                           height_ratios=(1.2, 1), layout="constrained")
    ax[0].plot(rho, L**3*g_int, color=COLORS[0], lw=1.4, label="Integral")
    ax[0].plot(rho[::10], L**3*g_img[::10], ls="none", marker="o", ms=3.5,
               mfc="white", mec=COLORS[1], label="Image series")
    ax[0].axhline(0, color="0.5", lw=.7)
    ax[0].set_ylabel(r"$L^3\mathcal{G}_{BB,zz}^{\mathrm{sc}}$")
    ax[0].legend(frameon=False, fontsize=FIGURE_LEGENDSIZE)
    ax[0].set_title(r"Static response, $r=0.65$", loc="right", fontsize=8.5, pad=10)
    ax[1].plot(rho, data[:, 1], lw=1.3)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("Relative difference")
    ax[1].set_xlabel(r"$\rho$ ($\mu$m)")
    for j, a in enumerate(ax):
        panel_label(a, f"({chr(97+j)})")
    style_axes(ax)
    savefig(fig, outdir, "FigA1_static_limit_check")


def propagating_green(rho, B, projection="zz", n_points=400):
    """Full propagating interval at the midplane, with q=k0*L*sqrt(1-u^2)."""
    if not (zA == zB == L/2):
        raise ValueError("The propagating diagnostic assumes both probes at the midplane")
    u, weights = np.polynomial.legendre.leggauss(n_points)
    u, weights = (u+1)/2, weights/2
    a = k0*L
    q = a*np.sqrt(1-u*u)
    kap_L = -1j*a*u
    v = r_te_slab(q/L, B)*np.exp(-kap_L)
    if projection == "zz":
        paths = 2*v/(1-v)
        prefactor = 1j*a**3*(1-u*u)/(4*np.pi*L**3)
    elif projection == "perp_avg":
        paths = 2*v/(1+v)
        prefactor = -1j*a**3*u*u/(8*np.pi*L**3)
    else:
        raise ValueError("projection must be 'zz' or 'perp_avg'")
    return np.sum(weights*prefactor*paths*j0(q*rho/L))


def validate_numerical_details():
    """Reproduce Appendix D at the nine stated bias/separation combinations."""
    points = [(B, rho) for B in BVALS_3 for rho in (0., 3e-6, 8e-6)]
    report = {
        "test_bias_mT": [8.0, 16.5, 40.0],
        "test_separation_um": [0.0, 3.0, 8.0],
        "q_min": 1e-4, "q_max": 80.0, "reference_grid_points": 24000,
        "grid_relative_differences": {}, "propagating_relative_contributions": {},
    }
    def components(values):
        values = np.asarray(values)
        return np.stack([values.real, values.imag])
    for projection in ("zz", "perp_avg"):
        reference = components([green_scalar(rho, B, projection=projection, nq=24000)
                                for B, rho in points])
        if np.any(reference == 0):
            raise RuntimeError("A relative grid error is undefined at a zero reference")
        errors = {}
        for n in (6000, 12000):
            values = components([green_scalar(rho, B, projection=projection, nq=n)
                                 for B, rho in points])
            errors[str(n)] = float(np.max(abs((values-reference)/reference)))
        tolerance = 2e-7 if projection == "zz" else 3e-3
        if not (errors["12000"] < errors["6000"] < tolerance):
            raise RuntimeError(f"Grid convergence failed for {projection}: {errors}")
        report["grid_relative_differences"][projection] = errors
        baseline = components([green_scalar(rho, B, projection=projection)
                               for B, rho in points])
        prop_values, contributions = [], {}
        for n in (200, 400, 800):
            values = np.asarray([propagating_green(rho, B, projection, n)
                                 for B, rho in points])
            prop_values.append(values)
            contributions[str(n)] = float(np.max(abs(components(values)/baseline)))
        for values in prop_values[:-1]:
            if not np.allclose(values, prop_values[-1], rtol=1e-9, atol=0):
                raise RuntimeError(f"Propagating quadrature failed to converge: {projection}")
        report["propagating_relative_contributions"][projection] = contributions
        print(f"  {projection} grid relative differences: {errors}")
        print(f"  {projection} propagating relative contributions: {contributions}")
    free = float(free_exchange_mHz(3e-6))
    correction = float(omega_mHz(green_scalar(3e-6, B_on)))
    report["exchange_at_3um_16p5mT_mHz"] = {
        "free_space": free, "cavity_correction": correction, "total": free+correction,
    }
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the numerical manuscript figures and validate the underlying data."
    )
    parser.add_argument("--recompute", action="store_true",
                        help="recompute data from physics and overwrite CSV files")
    parser.add_argument("--outdir", type=Path, default=Path("figures"),
                        help="output directory for PDF/PNG figures (default: figures/)")
    parser.add_argument("--report", action="store_true",
                        help="also evaluate the cavity-induced linewidth ratio quoted in the abstract")
    parser.add_argument(
        "--validate", action="store_true",
        help="recompute all data in memory, compare with the CSV files, and run static-limit checks",
    )
    parser.add_argument("--validation-json", type=Path,
                        help="write numerical diagnostics to JSON (requires --validate)")
    args = parser.parse_args()
    if args.validation_json is not None and not args.validate:
        parser.error("--validation-json requires --validate")

    here    = Path(__file__).parent
    csv_fig3  = here / "data_fig3.csv"
    csv_fig4  = here / "data_fig4.csv"
    csv_figA1 = here / "data_figA1.csv"
    csv_figA2 = here / "data_figA2.csv"

    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.recompute:
        print("Computing data (this may take a minute) ...")
        d3  = compute_kernel_data("zz");       save_csv(d3,  csv_fig3,  HDR_KERNEL)
        d4  = compute_data_fig4();             save_csv(d4,  csv_fig4,  HDR_FIG4)
        dA1 = compute_data_figA1();            save_csv(dA1, csv_figA1, HDR_FIGA1)
        dA2 = compute_kernel_data("perp_avg"); save_csv(dA2, csv_figA2, HDR_KERNEL)
    else:
        missing = [p for p in (csv_fig3, csv_fig4, csv_figA1, csv_figA2) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing CSV files: " + ", ".join(p.name for p in missing) +
                "\nRun with --recompute to generate them.")
        d3  = load_csv(csv_fig3)
        d4  = load_csv(csv_fig4)
        dA1 = load_csv(csv_figA1)
        dA2 = load_csv(csv_figA2)

    if args.validate:
        fresh = {
            "data_fig3.csv": compute_kernel_data("zz"),
            "data_fig4.csv": compute_data_fig4(),
            "data_figA1.csv": compute_data_figA1(),
            "data_figA2.csv": compute_kernel_data("perp_avg"),
        }
        stored = {
            "data_fig3.csv": d3,
            "data_fig4.csv": d4,
            "data_figA1.csv": dA1,
            "data_figA2.csv": dA2,
        }
        print("Validation report:")
        for name, values in fresh.items():
            max_abs = float(np.max(np.abs(values - stored[name])))
            if not np.allclose(values, stored[name], rtol=5.0e-12, atol=1.0e-15):
                raise RuntimeError(f"{name} does not match recomputed data (max abs diff {max_abs:.3e})")
            print(f"  {name}: matches recomputation (max abs diff {max_abs:.3e})")
        for projection in ("zz", "perp_avg"):
            rel = static_projection_check(projection)
            if not np.isfinite(rel) or rel >= 1.0e-6:
                raise RuntimeError(
                    f"static {projection} projection check failed (max relative difference {rel:.3e})"
                )
            print(f"  static {projection}: max relative difference {rel:.3e}")

    if args.validate:
        diagnostics = validate_numerical_details()
        if args.validation_json is not None:
            args.validation_json.parent.mkdir(parents=True, exist_ok=True)
            args.validation_json.write_text(json.dumps(diagnostics, indent=2) + "\n")

    print(f"Writing figures to {args.outdir.resolve()} ...")
    make_fig2(args.outdir)
    make_kernel_pair(d3, args.outdir, axis="zz",
                     filename="Fig3_single_spin_kernels")
    make_fig4(d4, args.outdir)
    make_figA1(dA1, args.outdir)
    make_kernel_pair(dA2, args.outdir, axis="xyavg",
                     filename="FigA2_transverse_projection",
                     note=r"transverse avg. $(xx+yy)/2$")

    if args.report:
        # green_scalar returns only the scattering part, with the free-space
        # tensor already subtracted. Its rho -> 0 limit is the finite
        # cavity-induced local correction rather than a bare local linewidth.
        rho_ref = 3.0e-6
        g12 = gamma_mHz(green_scalar(rho_ref, B_on))
        g11 = gamma_mHz(green_scalar(0.0, B_on))
        print(f"At B = {B_on * 1e3:.1f} mT, rho = {rho_ref * 1e6:.1f} um:")
        print(f"  delta gamma_12 / 2pi    = {g12:.3f} mHz")
        print(f"  delta gamma_11 / 2pi    = {g11:.3f} mHz")
        print(f"  cavity-induced linewidth ratio = {2.0 * abs(g12) / g11:.3f}")

    print("Done.")


if __name__ == "__main__":
    main()
