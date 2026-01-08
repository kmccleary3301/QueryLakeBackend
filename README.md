# QueryLake Backend

This is a FastAPI and Ray backend for QueryLake.
This repo now recommends using **`uv`** for Python environments + lockfiles.

You must have CUDA installed (and an appropriate NVIDIA driver) to run local models.
We recommend the [Lambda Stack](https://lambdalabs.com/lambda-stack-deep-learning-software).

## Install (recommended: `uv`)

Create a virtualenv and sync locked dependencies:

```
uv venv --python 3.12
uv sync
```

Optional extras:
- `uv sync --extra cli` (enables `setup.py` CLI helpers)
- `uv sync --extra inference-hf` (local HF/torch inference helpers)
- `uv sync --extra ocr` (OCR stack: Marker/Surya + OCRmyPDF)
- `uv sync --extra dev` (pytest tooling)

> Note: We intentionally keep **vLLM** as a separate runtime in production (run it as an upstream service and let QueryLake talk to it over HTTP). Use the `vllm` extra only for experiments.

## Install (legacy: conda + requirements.txt)

This is no longer the recommended path, but is kept for compatibility:

```bash
conda create --name QueryLake python=3.10
conda activate QueryLake
pip install -r requirements.txt
```

## ExllamaV2

One of the dependencies installed is `exllamav2`, however this occassionaly raises issues to the build. To safely install it, you should build from source by cloning it doing the following:

```bash
git clone https://github.com/turboderp/exllamav2
cd exllamav2
pip install -r requirements.txt
pip install .
cd ../
rm -rf exllamav2
```

## Tesseract
We currently support tesseract for OCR.
This requires apt installing tesseract like so:

```bash
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
```

## Database

The database is a ParadeDB container. To initialize it, you must have docker and docker-compose installed (use [these instructions](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)).
Once these are installed, you can run the following to start or completely reset the database:

```bash
./restart_database.sh
```

## Setup

To set up your models, run the `setup.py` CLI like so and follow the instructions:

```bash
python setup.py
```

I recommend using the presets for now, as custom model additions are under development.

## Start

I recommend starting a head node for ray clusters first. This initiates the ray dashboard, and may make it easier to connect serve deployments in the future. you can do so as follows:


```bash
ray start --head --port=6379 --dashboard-host 0.0.0.0
```

To start the server, run 

```bash
serve run server:deployment
```

Server settings are generated in `config.json`.
The file can be modified to your preferred settings.
