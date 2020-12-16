

tests: FORCE
	PYTHONPATH=. pytest -ra -q -v tests

codestyle:
	# general check
	black --check batteries
	# python code quality check
	flake8 \
		--max-line-length 120 \
		--ignore=Q000,D100,D205,D212,D400,D415,W605 \
		batteries

clear:
	rm -rf .pytest_cache
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete


ddp-example:
	PYTHONPATH=. python examples/ddp/experiment.py


FORCE: