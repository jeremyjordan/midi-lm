bootstrap:
	pyenv virtualenv 3.11 midi-lm
	pyenv local midi-lm
	python -m pip install pip-tools

requirements:
	pip-compile -o requirements.txt --resolver=backtracking pyproject.toml -v
	pip-compile --extra dev --extra app -o requirements-dev.txt --resolver=backtracking pyproject.toml -v

install:
	pip install -r requirements-dev.txt
	pip install -e .

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
