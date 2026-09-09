# Spatial Autocorrelation Analysis

A self-contained example of measuring **spatial autocorrelation** — the degree to
which nearby locations tend to have similar values — on 2-D point data using
**Moran's *I*** and **Local Indicators of Spatial Association (LISA)**.

The notebook generates synthetic spatial patterns with known structure, so the
statistics can be checked against ground truth, and walks through the full
workflow from raw points to an interpreted cluster map.

## What it does

1. **Generate synthetic datasets** — six point patterns (random, several gradients,
   and dispersed layouts) so the analysis can be run against known ground truth.
2. **Choose a neighborhood size *k*** — selected with the elbow method on k-means WCSS
   to parameterize a k-nearest-neighbors spatial weights matrix.
3. **Global Moran's *I*** — a single statistic testing for autocorrelation across the
   whole dataset, with a permutation-based p-value.
4. **Local Moran's *I* (LISA)** — a per-point decomposition that classifies each
   observation as a cluster (High-High / Low-Low) or a spatial outlier
   (High-Low / Low-High).
5. **Visualization** — the input values, the LISA cluster map, local *I* values, and
   local significance.

## Methods at a glance

- **Global Moran's *I*** summarizes the overall spatial pattern in one number:
  positive → similar values cluster; negative → high and low values are interspersed;
  near zero → no spatial structure.
- **LISA** breaks that global measure down per observation, so you can see *where*
  clustering or dispersion occurs and how significant each location is.

## Getting started

```bash
pip install -r requirements.txt
jupyter notebook "Spatial Autocorrelation Analysis.ipynb"
```

Then run the cells top to bottom. The first cell also installs the requirements
automatically (`%pip install -r requirements.txt`), so the manual `pip install` step
above is optional. Each cell prints an `[OK]` message on completion, so you can
confirm it ran successfully as you go.

## Choosing a dataset

The final line of the dataset cell selects which synthetic pattern is analyzed.
Change it to explore how each statistic responds to a different spatial structure:

| Variable | Pattern |
| --- | --- |
| `df_rando` | Uniformly random quality values |
| `df_gradient1` | Smooth low → high gradient |
| `df_gradient2` | High near the origin, mixing outward |
| `df_gradient3` | Gradient with a high value seeded near each pair of low values |
| `df_dispersed` | High and low values interleaved by quadrant |
| `df_dispersed_eq` | Same as above, on a perfectly equidistant grid *(default)* |

## Requirements

Python 3.9+ and the packages in [`requirements.txt`](requirements.txt): pandas,
NumPy, matplotlib, scikit-learn, and the [PySAL](https://pysal.org/) stack
(`libpysal`, `esda`).

## Notes

All data in this notebook is synthetic and generated in-notebook; no external data
files are required.
