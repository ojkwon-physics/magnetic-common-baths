# Finite-frequency magnetic common baths in ferromagnetic planar cavities

Reproduction code and data for O. J. Kwon, "Finite-frequency magnetic
common baths in ferromagnetic planar cavities."

## Contents

- `reproduce_figures.py` — regenerates the five data-bearing manuscript figures, optionally
  recomputes their underlying data, and runs numerical consistency checks.
- `validation_results.json` — numerical diagnostics from the validated release.
- `data_fig3.csv`, `data_fig4.csv`, `data_figA1.csv`, `data_figA2.csv` —
  precomputed data underlying the figures.
- `requirements.txt` — Python dependencies.

## Usage

Run these commands from the repository directory with Python 3.10 or later.
Install the dependencies first:

```
python -m pip install -r requirements.txt
```

```
python reproduce_figures.py             # load precomputed CSVs, generate figures
python reproduce_figures.py --recompute # recompute from physics, update CSVs
python reproduce_figures.py --outdir DIR # write figures to DIR (default: figures/)
python reproduce_figures.py --report    # print the cavity-induced linewidth ratio
python reproduce_figures.py --validate --validation-json validation_results.json
```

The default command writes both PDF and PNG versions of
`Fig2_reflection_map`, `Fig3_single_spin_kernels`,
`Fig4_bright_observables`, `FigA1_static_limit_check`, and
`FigA2_transverse_projection`.

The output names correspond to the final manuscript as follows:

| Manuscript figure | Output stem | Data source |
| --- | --- | --- |
| Fig. 2 | `Fig2_reflection_map` | Calculated directly from the reflection amplitudes |
| Fig. 3 | `Fig3_single_spin_kernels` | `data_fig3.csv`; analytic free-space exchange added for panel (c) |
| Fig. 4 | `Fig4_bright_observables` | `data_fig4.csv` |
| Fig. 5 | `FigA1_static_limit_check` | `data_figA1.csv`; integral and image series evaluated for panel (a) |
| Fig. 6 | `FigA2_transverse_projection` | `data_figA2.csv` |

The `FigA1` and `FigA2` filenames are retained for compatibility; they refer
to Figs. 5 and 6 in the manuscript.

Figure 1 is a conceptual schematic and contains no numerical data, so it is
not included in the reproduction script.

The main separation plot includes the total exchange, computed by adding the
analytic free-space contribution to the stored cavity correction. The static
comparison plots both representations and retains the stored relative differences.
No CSV values are changed by plotting or validation.

Validation compares all four CSV files with fresh calculations, checks the static
image series, and reproduces the Appendix D grid and propagating-wave interval checks.
Grid errors compare the real and imaginary parts separately at the nine stated
bias/separation combinations against 24,000 grid points. Propagating contributions
are normalized to the corresponding parts of the production scattering integral;
they are not bounds on the total integration error. The diagnostic also records
the free-space, cavity-induced, and total exchange at 3 micrometres and 16.5 mT.

The signed cavity correction can change sign while the total exchange remains
positive. Figure 4 plots the exchange correction itself; the correction to the
mode frequency difference is twice this value.

## Model and plotted quantities

The calculation uses a phenomenological scalar permeability and TE reflection.
The main Green-tensor projection is `zz`; the transverse TE result is the
per-component average `(xx+yy)/2`. It does not include the TM contribution.
Figure 2 labels the reflection amplitude as `r_s`, as in the manuscript.

The `--report` ratio is the cavity-induced linewidth difference
`2*abs(delta_gamma_12)` divided by the local decay correction `delta_gamma_11`.
It is approximately 0.66 at a separation of 3 micrometres and a bias of 16.5 mT.
The dotted lines in Fig. 4 mark this ON bias, near the film resonance.
The ensemble curves assume uniform phases and couplings. Resolving the two
resonances requires comparing their total frequency separation with their
individual linewidths.
