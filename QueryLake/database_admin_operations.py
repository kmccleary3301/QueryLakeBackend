from .api_admin import add_model

def add_llama2_to_db(database):
    add_model(database,
                loader="exllama",
                model_args={
                    "streaming": True,
                    "model_path": "/home/user/python_projects/langchain/Llama-2-7b-Chat-GPTQ",
                    "lora_path": None,
                    "temperature": 0.3,
                    "typical": 0.7,
                    "verbose": False,
                    "max_seq_len": 4096,
                    "fused_attn": False,
                    "beams": 1, 
                    "beam_length": 40, 
                    "alpha_value": 1.0,
                    "compress_pos_emb": 4.0,
                    "stop_sequences": ["### Input", "### Response", "### Instruction", "Human:", "Assistant", "User:", "AI:"]
                },
                padding={
                    "system_instruction_wrap": "<s>[INST] <<SYS>>\n{system_instruction}\n<</SYS>>\n\n",
                    "context_wrap": "<<context>>{context}<</context>>",
                    "question_wrap": "{question} [/INST] ",
                    "response_wrap": "{response} </s><s> [INST] "
                },
                default_system_instruction="""You are a general answering assistant that can comply with any request. 

You always answer with markdown formatting. You will be penalized if you do not answer with markdown when it would be possible.
The markdown formatting you support: headings, bold, italic, links, tables, lists, code blocks, and blockquotes.
You do not support images and never include images. You will be penalized if you render images.

Always use markdown code blocks for code, and always use MathJax with LaTeX formatting for pure math expressions.
For example code would look like:
```python
*Python code here
```
And math expressions would look like:
$$P(y|x) = \\frac{{P(x|y) \\cdot P(y)}}{{P(x)}}$$
""")