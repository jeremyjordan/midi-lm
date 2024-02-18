bootstrap:
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv venv

requirements:
	uv pip compile pyproject.toml -o requirements.txt
	uv pip compile --extra dev --extra app pyproject.toml -o requirements-dev.txt

install:
	uv pip install -r requirements-dev.txt
	uv pip install -e .

test:
	pytest .
	find . -name '.coverage*' -exec rm -f {} +

format:
	ruff check .
	ruff format .

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '.coverage*' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
