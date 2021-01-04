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

If CUDA:
Install latest cuda-toolkit

```shell
sudo apt-get install cuda-toolkit-11-2

```

Explicitly set `CUDAHOME`

```shell
CUDAHOME=/usr/local/cuda-11.2 pip install -r requirements.txt
```

If you already accidentally installed `implicit`:

```shell
CUDAHOME=/usr/local/cuda-11.2 pip install --upgrade --force-reinstall --no-cache-dir -r requirements.txt
```

## Config

```shell
cp config_example.py config.py
```

## Running

```shell
python -m app.extract
```

## Development

```shell
pip install pre-commit black isort flake8
pre-commit install
```
