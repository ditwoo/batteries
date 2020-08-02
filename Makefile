

tests: FORCE
	PYTHONPATH=. pytest -ra -q -v tests

clear:
	rm -rf .pytest_cache
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

FORCE: