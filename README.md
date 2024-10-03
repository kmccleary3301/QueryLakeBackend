# QueryLake Backend

This is a FastAPI and Ray backend for QueryLake.
Run the following commands to set up a conda environment.
You must have CUDA installed to run models.
We recommend the [Lambda Stack](https://lambdalabs.com/lambda-stack-deep-learning-software)

```
conda create --name QueryLake python=3.10
conda activate QueryLake
```

After this, install pytorch using [this webpage](https://pytorch.org/).
Then continue with the following command.

```
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

The database is a ParadeDB container. To initialize it, you must have docker and docker-compose installed.
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

To

To start the server, run 

```bash
serve run server:deployment
```

Server settings are generated in `config.json`.
The file can be modified to your preferred settings.