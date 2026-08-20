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
  python reproduce_figures.py --report      # print the scattering-sector ratio
"""

from __future__ import annotations

from pathlib import Path
import argparse
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

COLUMNWIDTH_FIGSIZE = (3.40, 1.82)
COLUMN_LABELSIZE = 7.0
COLUMN_TICKSIZE = 6.2
COLUMN_LEGENDSIZE = 5.9
COLUMN_TITLESIZE = 5.7
COLUMN_PARAMSIZE = 5.3
COLUMN_PANELSIZE = 7.4
COLUMN_LINEWIDTH = 1.00
COLUMN_AXIS_LINEWIDTH = 0.75
COLUMN_REF_LINEWIDTH = 0.85

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
def axis_label(axis: str) -> str:
    """Return compact text for the projection axis used in annotations."""
    return "(xx+yy)/2" if axis == "xyavg" else axis


def parameter_label_column(axis: str = "zz", t: float = film_t) -> str:
    """Return a short parameter label for a two-panel figure at column width."""
    if axis == "xyavg":
        return f"t={t*1e9:.0f} nm; sub=metal"
    return f"t={t*1e9:.0f} nm; sub=metal; proj={axis_label(axis)}"


def panel_label(ax: mpl.axes.Axes, label: str, fontsize: float = 10.0) -> None:
    """Place a panel label in axes coordinates."""
    ax.text(0.055, 0.93, label, transform=ax.transAxes,
            ha="left", va="top", fontweight="bold", fontsize=fontsize)


def format_columnwidth_two_panel(ax) -> None:
    """Apply final-print typography for a two-panel figure used at column width."""
    for a in np.ravel(ax):
        a.xaxis.label.set_size(COLUMN_LABELSIZE)
        a.yaxis.label.set_size(COLUMN_LABELSIZE)
        a.tick_params(axis="both", which="major",
                      labelsize=COLUMN_TICKSIZE,
                      width=COLUMN_AXIS_LINEWIDTH, length=2.35)
        a.tick_params(axis="both", which="minor", width=0.60, length=1.45)
        for spine in a.spines.values():
            spine.set_linewidth(COLUMN_AXIS_LINEWIDTH)


def plot_signed(ax: mpl.axes.Axes, x: np.ndarray, y: np.ndarray,
                label: str, lw: float = 1.25) -> None:
    """Plot signed data with solid positive and dashed negative segments."""
    y = np.asarray(y, dtype=float)
    pos = y >= 0.0
    neg = y < 0.0
    if np.any(pos):
        line, = ax.plot(x, np.where(pos, y, np.nan), lw=lw, label=label)
        color = line.get_color()
        if np.any(neg):
            ax.plot(x, np.where(neg, y, np.nan), lw=lw, ls="--", color=color)
    else:
        ax.plot(x, y, lw=lw, ls="--", label=label)


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

    fig, ax = plt.subplots(1, 3, figsize=(8.1, 2.8),
                           width_ratios=(1.0, 1.0, 1.05), layout="constrained")

    im0 = ax[0].pcolormesh(Bvals * 1e3, qvals, log_slab, shading="gouraud", vmin=vmin, vmax=vmax)
    ax[1].pcolormesh(Bvals * 1e3, qvals, log_inf, shading="gouraud", vmin=vmin, vmax=vmax)

    for a in ax[:2]:
        a.set_yscale("log")
        a.set_xlabel(r"$B$ (mT)")
        a.set_ylabel(r"$k_\perp L$")
        a.tick_params(axis="both", which="major", labelsize=8.8)

    panel_label(ax[0], "(a)", fontsize=9.8)
    panel_label(ax[1], "(b)", fontsize=9.8)

    text_bbox = dict(facecolor="white", alpha=0.68, edgecolor="none", pad=1.3)
    ax[0].text(0.95, 0.055, r"finite film", transform=ax[0].transAxes,
               ha="right", va="bottom", fontsize=8.4, bbox=text_bbox)
    ax[1].text(0.95, 0.055, r"$t\to\infty$", transform=ax[1].transAxes,
               ha="right", va="bottom", fontsize=8.4, bbox=text_bbox)

    cb = fig.colorbar(im0, ax=[ax[0], ax[1]], shrink=0.91, pad=0.018)
    cb.set_label(r"$\log_{10}|r_{\mathrm{TE}}|$", fontsize=9.0)
    cb.ax.tick_params(labelsize=8.3)

    finite_cut = np.abs(r_te_slab(kk, B_on, half_space=False))
    half_cut = np.abs(r_te_slab(kk, B_on, half_space=True))

    ax[2].plot(qvals, finite_cut, lw=1.45, label=r"finite film")
    ax[2].plot(qvals, half_cut, ls="--", lw=1.45, label=r"$t\to\infty$")
    ax[2].set_xscale("log")

    yline = np.r_[finite_cut, half_cut]
    ax[2].set_ylim(max(0.0, 0.983 * np.nanmin(yline)), 1.055 * np.nanmax(yline))
    ax[2].set_xlabel(r"$k_\perp L$")
    ax[2].set_ylabel(r"$|r_{\mathrm{TE}}|$")
    ax[2].tick_params(axis="both", which="major", labelsize=8.8)

    ax[2].set_title(r"$B=16.5$ mT", fontsize=8.4, loc="left", pad=5)
    panel_label(ax[2], "(c)", fontsize=9.8)

    ax[2].legend(frameon=True, loc="upper left", bbox_to_anchor=(0.06, 0.84),
                 fontsize=8.0, handlelength=1.45, labelspacing=0.25,
                 handletextpad=0.45, borderpad=0.25, borderaxespad=0.0,
                 framealpha=0.72, edgecolor="none", facecolor="white")

    savefig(fig, outdir, "Fig2_reflection_map")


def make_kernel_pair(data: np.ndarray, outdir: Path, axis: str,
                     filename: str, note: str | None = None) -> None:
    """Generate a 1x2 signed decay/exchange kernel figure for a selected projection."""
    rho_um = data[:, 0]
    gam = [data[:, 1], data[:, 2], data[:, 3]]
    om = [data[:, 4], data[:, 5], data[:, 6]]

    fig, ax = plt.subplots(1, 2, figsize=COLUMNWIDTH_FIGSIZE, layout="constrained")

    for i, label in enumerate(LABELS_3):
        plot_signed(ax[0], rho_um, gam[i], label, lw=COLUMN_LINEWIDTH)
        ax[1].plot(rho_um, om[i], lw=COLUMN_LINEWIDTH, label=label)

    maxabs = max(np.max(np.abs(g)) for g in gam)
    linthresh = max(maxabs * 1e-1, 1e-2)
    ax[0].set_yscale("symlog", linthresh=linthresh)
    ax[0].axhline(0.0, lw=COLUMN_AXIS_LINEWIDTH, color=REF_COLOR)

    ax[0].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax[0].tick_params(axis="y", which="minor", left=False, right=False)

    ax[0].set_xlabel(r"$\rho$ ($\mu$m)")
    if axis == "zz":
        ax[0].set_ylabel(r"$\delta\gamma_{12}/2\pi$ (mHz)")
        ax[1].set_ylabel(r"$\Omega_{12}^{\mathrm{sc}}/2\pi$ (mHz)")
    else:
        ax[0].set_ylabel(r"$\delta\gamma_{12}^{\perp}/2\pi$ (mHz)")
        ax[1].set_ylabel(r"$\Omega_{12}^{\mathrm{sc},\perp}/2\pi$ (mHz)")

    if note:
        ax[0].set_title(note, fontsize=COLUMN_TITLESIZE, loc="left", pad=3)

    panel_label(ax[0], "(a)", fontsize=COLUMN_PANELSIZE)

    ax[1].axhline(0.0, lw=COLUMN_AXIS_LINEWIDTH, color=REF_COLOR)
    ax[1].set_xlabel(r"$\rho$ ($\mu$m)")
    ax[1].legend(frameon=False, fontsize=COLUMN_LEGENDSIZE)
    ax[1].set_title(parameter_label_column(axis=axis),
                    fontsize=COLUMN_PARAMSIZE, loc="right", pad=3)
    panel_label(ax[1], "(b)", fontsize=COLUMN_PANELSIZE)

    format_columnwidth_two_panel(ax)
    savefig(fig, outdir, filename)


def make_fig4(data: np.ndarray, outdir: Path) -> None:
    """Generate ideal bright-mode N_eff scaling curves."""
    B_mT = data[:, 0]
    gam_mHz = data[:, 1]
    om_mHz = data[:, 2]

    fig, ax = plt.subplots(1, 2, figsize=COLUMNWIDTH_FIGSIZE, layout="constrained")
    for N in (1000, 5000):
        ax[0].plot(B_mT, 2.0 * N * gam_mHz / 1e3, lw=COLUMN_LINEWIDTH,
                   label=rf"$N_{{\mathrm{{eff}}}}={N}$")
        ax[1].plot(B_mT, N * om_mHz / 1e3, lw=COLUMN_LINEWIDTH,
                   label=rf"$N_{{\mathrm{{eff}}}}={N}$")

    maxabs = np.max(np.abs(2.0 * 5000 * gam_mHz / 1e3))
    ax[0].set_yscale("symlog", linthresh=max(maxabs * 1e-1, 1e-2))
    ax[0].axhline(0.0, lw=COLUMN_AXIS_LINEWIDTH, color=REF_COLOR)
    ax[0].axhline(1.0, ls=":", lw=COLUMN_REF_LINEWIDTH, color=REF_COLOR)
    ax[0].axhline(-1.0, ls=":", lw=COLUMN_REF_LINEWIDTH, color=REF_COLOR)
    ax[0].set_xlabel(r"$B$ (mT)")
    ax[0].set_ylabel(r"$2\delta\gamma_{AB}^{\mathrm{br}}/2\pi$ (Hz)")
    panel_label(ax[0], "(a)", fontsize=COLUMN_PANELSIZE)

    ax[1].axhline(0.0, lw=COLUMN_AXIS_LINEWIDTH, color=REF_COLOR)
    ax[1].axhline(1.0, ls=":", lw=COLUMN_REF_LINEWIDTH, color=REF_COLOR)
    ax[1].axhline(-1.0, ls=":", lw=COLUMN_REF_LINEWIDTH, color=REF_COLOR)
    ax[1].set_xlabel(r"$B$ (mT)")
    ax[1].set_ylabel(r"$\Omega_{AB}^{\mathrm{br,sc}}/2\pi$ (Hz)")
    ax[1].set_title(parameter_label_column(axis="zz"),
                    fontsize=COLUMN_PARAMSIZE, loc="right", pad=3)
    ax[1].legend(frameon=False, fontsize=COLUMN_LEGENDSIZE)
    panel_label(ax[1], "(b)", fontsize=COLUMN_PANELSIZE)

    format_columnwidth_two_panel(ax)
    savefig(fig, outdir, "Fig4_bright_observables")


def make_figA1(data: np.ndarray, outdir: Path) -> None:
    """Generate static image-series versus q-integral consistency check."""
    rho_um = data[:, 0]
    rel = data[:, 1]
    r_const = 0.65

    fig, ax = plt.subplots(figsize=(3.40, 2.12), layout="constrained")
    ax.plot(rho_um, rel, lw=1.15)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\rho$ ($\mu$m)")
    ax.set_ylabel("relative difference")
    ax.set_title(r"projection: zz", fontsize=7.1, loc="left", pad=3)
    ax.set_title(rf"static limit check ($r={r_const}$)", fontsize=6.7, loc="right", pad=3)
    ax.tick_params(axis="both", which="major", labelsize=7.0, width=0.8, length=2.7)
    ax.tick_params(axis="both", which="minor", width=0.65, length=1.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.xaxis.label.set_size(8.0)
    ax.yaxis.label.set_size(8.0)
    panel_label(ax, "(a)", fontsize=8.2)
    savefig(fig, outdir, "FigA1_static_limit_check")

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
                        help="also evaluate the scattering-sector ratio quoted in the abstract")
    parser.add_argument(
        "--validate", action="store_true",
        help="recompute all data in memory, compare with the CSV files, and run static-limit checks",
    )
    args = parser.parse_args()

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
        print(f"  scattering-sector ratio = {2.0 * abs(g12) / g11:.3f}")

    print("Done.")


if __name__ == "__main__":
    main()
