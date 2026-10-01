# Rebuild everything from scratch: about 15 minutes on a laptop.
PY ?= python3

all: ecg data model unit demo diagram

ecg:      ## 1. potassium-from-ECG sensor (calibration cohort)
	$(PY) scripts/train_ecg_estimator.py
data:     ## 2. simulate train/test units, record nightly features + labels
	$(PY) scripts/build_datasets.py
model:    ## 3. risk model + ablation study
	$(PY) scripts/train_risk_model.py
unit:     ## 4. scheduling simulation, 6 replicate units
	$(PY) scripts/simulate_unit.py --replicates 6
demo:     ## 5. frozen Tuesday-night unit for the dashboard
	$(PY) scripts/build_demo.py
diagram:
	$(PY) scripts/draw_architecture.py
app:
	streamlit run app/dashboard.py
test:
	$(PY) -m pytest -q tests

.PHONY: all ecg data model unit demo diagram app test
