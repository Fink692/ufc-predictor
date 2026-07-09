# Contributing

Thanks for taking a look at UFC Predictor.

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Checks

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -v
python -m compileall -q src tests
```

## Data And Model Artifacts

Do not commit generated files from `data/`, `models/`, or `reports/`. They are reproducible from public data and ignored intentionally.

## Modeling Rules

- Keep features leakage-safe.
- Use chronological validation when reporting model quality.
- Compare against simple baselines.
- Document any new data source and its limitations.
