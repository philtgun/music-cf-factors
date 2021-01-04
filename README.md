# Music CF factors

Extracting collaborative-filtering item factors from music

## Requirements

Python 3.7+

```shell
python3.x -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Config

```shell
mv config_example.py config.py
```

## Development
```shell
pip install pre-commit black isort flake8
```

## Run
```shell
python -m app.extract song_factors.npy
```
