test:
	venv/bin/python -m pytest tests/test_new_algorithms.py -v --tb=short

show:
	venv/bin/python examples/cli_demo.py --dataset moons --samples 500 --no-plot