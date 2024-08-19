# Voltage clamp model with NaV current project

This is a repository to reproduce results for the work "_Resolving artefacts in voltage-clamp experiments with computational modelling: an application to fast sodium current recordings_".

### Dependencies
To run the code, run pip install -r requirements.txt to install all the necessary dependencies. Python >3.6 is required (tested on Python 3.11).

### Content

- `data`: Experimental data.
- `methods`: Main python codes.
- `models`: Myokit models.
- `protocols`: Voltage clamp protocols.
- `src`: Other source code for exploring experimental artefact effects.
- `fit.py`: Run model fitting.
- `figure-plot.py`: Produce figure for fitted results.
- `figure-plot-diff-models.py`: Produce figure for comparing two fitted models.

### Data source
<https://app.box.com/folder/156288669561>

### Units
- time in [ms]
- voltage in [mV]
- current in [pA]
- capacitance in [pF]
- resistance in [GOhm]
