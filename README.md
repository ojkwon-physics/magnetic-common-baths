# Finite-frequency magnetic common baths in ferromagnetic planar cavities

Reproduction code and data for O. J. Kwon, "Finite-frequency magnetic
common baths in ferromagnetic planar cavities."

## Contents

- `reproduce_figures.py` — regenerates the five data-bearing manuscript figures, optionally
  recomputes their underlying data, and runs numerical consistency checks.
- `data_fig3.csv`, `data_fig4.csv`, `data_figA1.csv`, `data_figA2.csv` —
  precomputed data underlying the figures.
- `requirements.txt` — Python dependencies.

## Usage

```
python reproduce_figures.py             # load precomputed CSVs, generate figures
python reproduce_figures.py --recompute # recompute from physics, update CSVs
python reproduce_figures.py --outdir DIR # write figures to DIR (default: figures/)
python reproduce_figures.py --report    # print the scattering-sector ratio
python reproduce_figures.py --validate  # compare CSVs with recomputation and check static limits
```

The default command writes both PDF and PNG versions of
`Fig2_reflection_map`, `Fig3_single_spin_kernels`,
`Fig4_bright_observables`, `FigA1_static_limit_check`, and
`FigA2_transverse_projection`.

Figure 1 is a conceptual schematic and contains no numerical data, so it is
not included in the reproduction script.
