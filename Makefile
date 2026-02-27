.PHONY: demo test quality-baseline quality-gate

PYTHON ?= python

demo:
	$(PYTHON) scripts/run_demo.py

test:
	$(PYTHON) -m pytest

quality-baseline:
	$(PYTHON) scripts/field_quality_baseline.py

quality-gate:
	$(PYTHON) scripts/field_quality_baseline.py --gate
