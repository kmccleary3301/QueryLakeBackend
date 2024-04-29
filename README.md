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

To start the server, run `serve run server:deployment`
Server settings are in `config.json`, and the file can be modified to your preferred settings.