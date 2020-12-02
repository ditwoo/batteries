

tests: FORCE
	PYTHONPATH=. pytest -ra -q -v tests

codestyle:
	# general check
	black --check batteries
	# python code quality check
	flake8 \
		--count \
		--max-line-length 120 \
		--docstring-convention google \
		--ignore=Q000,D100,D205,D212,D415,W605 \
		--show-source \
		--statistics \
		batteries

clear:
	rm -rf .pytest_cache
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

FORCE: