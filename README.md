# QueryLake Backend

This is a FastAPI and Uvicorn backend for QueryLake.
Run the following commands to set up a conda environment.

```
conda create --name QueryLake python=3.9
conda activate QueryLake
pip install -r requirements.txt
python -m pip install git+https://github.com/kmccleary3301/exllama
```

The last one is important because we are using a fork of exllama that
adds compatibility with LangChain.

To start the server, run `python server.py`