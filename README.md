# Voltage clamp model with NaV current project

This is a repository to reproduce results for the work "_Resolving artefacts in voltage-clamp experiments with computational modelling: an application to fast sodium current recordings_".

### Dependencies
To run the code, run `pip install -r requirements.txt` to install all the necessary dependencies. Python >3.6 is required (tested on Python 3.9 and 3.11).

### Reproducing result figures
- Figure 3C: Change directory to `src` and run `python simulate-mc4auto-minimum.py nav`.
- Figure 4C: Change directory to `src` and run `python compensation-level-sweeps.py`.
- Figure 5C: Run `python figure-plot.py -m iyer -p NaIV_35C_80CP -d cell6 -l 90`.
- Figure 6A: Change directory to `src` and run `python averaging-issue.py`.
- Figure 6B: Change directory to `src` and run Jupyter notebook `mutant-issue.ipynb`.
- Figure 7: Run `python figure-plot-diff-models.py -m gray -p NaIV_35C_80CP -d cell9 -l 90`.

### Content

- `data`: Experimental data (see below for how to download this).
- `methods`: Main python codes.
- `models`: Myokit models.
- `protocols`: Voltage clamp protocols.
- `src`: Other source code for studying experimental artefact effects.
- `fit.py`: Run model fitting to produce results in `results`.

### Data source
Experimental data of this study may be downloaded from the following link: <https://doi.org/10.6084/m9.figshare.27193878>.

The whole dataset should be placed within this repository as `data` such that data can be loaded and read properly.

### Units
- time in [ms]
- voltage in [mV]
- current in [pA]
- capacitance in [pF]
- resistance in [GOhm]


### Acknowledging this work
If you publish any work based on the contents of this repository please cite ([CITATION file](CITATION)):

Chon Lok Lei, Alexander P. Clark, Michael Clerx, Siyu Wei, Meye Bloothooft, Teun P. de Boer, David J. Christini, Trine Krogh-Madsen, Gary R. Mirams.
(2024).
[Resolving artefacts in voltage-clamp experiments with computational modelling: an application to fast sodium current recordings](https://doi.org/10.1101/2024.07.23.604780).
bioRxiv, 2024.07.23.604780.
