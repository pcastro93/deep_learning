# Deep Learning

This project is for learning purposes only.

## Structure

- `deep_learning`: The main package containing the code for the project.
    - `app`: The application code.
        - `main.py`: The main script for running the application.
        - `models.py`: The models used in the application.
        - `linear_unit`: The linear unit module.
            - `synthetic_data.py`: Showcases the linear unit with synthetic data.
            - `ww2.py`: WW2 Kaggle Problem
    - `tests`: The tests for the project.
        - `test_models.py`: The tests for the models.
    - `README.md`: The README file for the project.

## Setup
- Create virtual environment:
```sh
uv venv
```

- Install dependencies:
```sh
uv sync
```

- Install the package in editable mode once (from the `deep_learning` folder):
```sh
uv pip install -e .
```

- Run whatever script
```sh
uv run python -m app.main
```
