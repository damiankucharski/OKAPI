# When does a fusion tree earn its complexity?

A research note from OKAPI and FusionBench, November 2026. The note follows three questions. Can a native tree express a useful rule? Can evolution find and retain it? Does choosing it on development data help on fresh observations?

[Read the illustrated PDF](study.pdf) · [Browse the evidence by question](evidence.md) · [Use the native fitting API](../../api/native-fitting.md)

## Which OKAPI won the comparison?

On 144 matched prediction banks covering 24 fixed scenarios, the regularized procedure that selected among three native trees on validation reached 0.21456 mean test Brier. Probability logistic stacking scored 0.24431. Deploying the GP candidate alone on those same banks scored 0.28419. Six scenario means favored stacking. Figure 1 gives the differences for every scenario and the tree families chosen on validation. Scores for the complete procedure require the archived split, search and refit settings. [Paired results and full protocol](https://github.com/damiankucharski/fusionbench/blob/5f9254e0c6e4559d7c0d19f28dc91641fba09d57/docs/ensemble-study/replacement-experiment/panel/README.md)

## Review decisions

| Component | Decision and evidence |
| --- | --- |
| Signed native head fitting | Exposed as an explicit helper with a caller-defined scalar objective. The fitter supports a variance-scaled weight penalty. On ten fresh banks per family across 19 families, the fixed penalty reduced the prespecified weighted Brier from 0.21524 to 0.20673. [Ablation](https://github.com/damiankucharski/fusionbench/blob/5f9254e0c6e4559d7c0d19f28dc91641fba09d57/docs/ensemble-study/pattern-iterations/README.md) |
| Archive feature selection | Exposed as a separate deterministic helper. It lets an already fitted head use nonlinear outputs of evolved programs. On the original 19-family IID suite, the flat and augmented candidates scored 0.21891 and 0.18248, respectively. [Family comparison](https://github.com/damiankucharski/fusionbench/blob/5f9254e0c6e4559d7c0d19f28dc91641fba09d57/docs/ensemble-study/native-pattern/README.md) |
| Joint discovery, validation and refit procedure | Kept in the research adapter. It specifies data splitting and a total fitting protocol. Using the two library functions alone does not reproduce its reported score. [Adapter](https://github.com/damiankucharski/fusionbench/blob/5f9254e0c6e4559d7c0d19f28dc91641fba09d57/fusionbench/methods/native_pattern.py) |
| Alternative training or replacement policies | Kept experimental. The global uncertainty penalty worsened full-suite test Brier. The certified replacement rule missed a supplied correct improvement in all 48 nonlinear signal banks. [Uncertainty penalty](https://github.com/damiankucharski/fusionbench/blob/5f9254e0c6e4559d7c0d19f28dc91641fba09d57/docs/ensemble-study/validation-curse/README.md) · [Power check](https://github.com/damiankucharski/fusionbench/blob/5f9254e0c6e4559d7c0d19f28dc91641fba09d57/docs/ensemble-study/replacement-experiment/controlled/README.md) |
| Operator retyping and structural identity | Retained for a separate review of the evolution engine. Retyping a node while preserving its children improved IID Brier by 0.01202 against a matched fourth mutation slot at 1,200 calls. Adding that slot also changes proposal frequencies. Repairing structural identities changed five of 48 search trajectories in the reported control and none of the selected winners. [Mutation slot comparison](https://github.com/damiankucharski/fusionbench/blob/5f9254e0c6e4559d7c0d19f28dc91641fba09d57/docs/ensemble-study/mutation-opportunity/README.md) · [Identity check](https://github.com/damiankucharski/fusionbench/blob/5f9254e0c6e4559d7c0d19f28dc91641fba09d57/docs/ensemble-study/search-regret/structural-key/README.md) |
| Alternate initialization, cross-class heads, compilation and prediction-bank diagnostics | Retained with their controlled experiments. Several show substantial gains on specific laws. They change separate algorithm or deployment contracts and should be reviewed on their own evidence. [Evidence index](evidence.md) |

## Regenerate the article

The figures and compiled PDF are checked in. For editing, download [study.tex](study.tex) and [mechanisms.tex](mechanisms.tex), or work in this directory in an OKAPI checkout. The sources include [Figure 1](figures/native-performance.pdf) and [Figure 2](figures/replacement-mechanisms.pdf). Run the following commands from this directory with a TeX Live installation containing pdfLaTeX, latexmk and Latin Modern.

```sh
mkdir -p build
SOURCE_DATE_EPOCH=1795824000 latexmk -pdf -halt-on-error -interaction=nonstopmode -outdir=build study.tex
cp build/study.pdf study.pdf
```

The [plotting script](make_figures.py) reads the archived CSVs through Git. A FusionBench clone containing revision `5f9254e0c6e4559d7c0d19f28dc91641fba09d57`, Python with NumPy, pandas and Matplotlib, and the two commands below suffice to regenerate the graphics. The checked-in figures were rendered with NumPy 2.3.5, pandas 2.2.3 and Matplotlib 3.10.8. Model training is unnecessary.

```sh
git clone --branch research/ensemble-mechanisms-stability https://github.com/damiankucharski/fusionbench.git /tmp/fusionbench-study
python make_figures.py --fusionbench /tmp/fusionbench-study --out figures
```

The [evidence index](evidence.md) links to the protocols, reproduction drivers, raw tables, literature and earlier research branches. Its first section pins the snapshots for reanalysis.
