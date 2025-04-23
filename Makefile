

tests: FORCE
	@ uv run pytest -ra -q -v

codestyle:
	@ uvx isort --line-length=120 --profile=black src && \
	uvx black --line-length=120 src

codestyle-check:
	@ uvx isort --line-length=120 --profile=black src && \
	uvx black --line-length=120 src && \
	uvx flake8 --max-line-length 120 --ignore=Q000,D100,D205,D212,D400,D415,W605 src

clear:
	rm -rf .pytest_cache; \
	find src -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete


# ddp-example:
# 	PYTHONPATH=. python examples/ddp/experiment.py


FORCE: