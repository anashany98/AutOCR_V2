.PHONY: demo test

PYTHON ?= python

demo:
	$(PYTHON) scripts/run_demo.py

test:
	$(PYTHON) -m pytest
