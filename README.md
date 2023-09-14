# QueryLake Backend

This is a FastAPI and Uvicorn backend for QueryLake.
Run the following commands to set up a conda environment.

```
conda create --name QueryLake python=3.9
conda activate QueryLake
```

After this, install pytorch using [this webpage](https://pytorch.org/).
Then continue with the following command.

```
pip install -r requirements.txt
```

To start the server, run `python server.py`

Please note, the server supports using either [llama.cpp](https://github.com/ggerganov/llama.cpp) or [exllama](https://github.com/turboderp/exllama) for loading models. Exllama is preferable for GPU loading, however llama.cpp is designed for loading onto Apple's M2 architecture.
Exllama only works with GPTQ model weights, and llama.cpp only works with GGUF model weights.
These are quantization formats, and if you are using a public model, you can likely find the quantized weights in your preferred format on Huggingface with a simple google search.

Server settings are in `config.json`, and the file can be modified accordingly to your preferred settings.