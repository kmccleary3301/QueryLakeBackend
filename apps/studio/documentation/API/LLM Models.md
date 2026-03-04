You can call an LLM using a normal post or get request using QueryLake authentication.


## Authentication
This endpoint supports three types of authentication. Only one is required for any request:
1. **API Key**
   ```json
   {"auth": {"api_key": "example_api_key"}}
   ```
2. **Username and Password**
   ```json
   {"auth": {"username": "example_username", "password": "example_password"}}
   ```
3. **OAuth2 Token**
   ```json
   {"auth": "oauth2_string"}
   ```

## Standard Call
For a static call, you would call it like so:
```python
import requests

response = requests.get(f"http://localhost:8000/api/llm", json={
    "auth": {"api_key": "sk-123456789"}, 
    "question": "What is the Riemann-Roch theorem?",
    "model_parameters": {
	    "temperature": 1.0
    }
})
result = response.raise_for_status().json()
```

Alternatively, you can provide chat history:
```python
import requests

response = requests.get(f"http://localhost:8000/api/llm", json={
    "auth": {"api_key": "sk-123456789"}, 
    "chat_history": [
	    {"role": "user", "content": "What is the Riemann-Roch theorem?"}
    ],
    "model_parameters": {
	    "temperature": 1.0
    }
})
result = response.raise_for_status().json()
```

#### Results
```json
{
    "success": true,
    "result": {
        "output": "The Riemann-Roch Theorem\n================--------\n\nThe Riemann-Roch theorem is a fundamental result in algebraic geometry that relates the dimension of the space of global sections of a line bundle on a compact Riemann surface to its degree.\n\nLet \\( X \\) be a compact Riemann surface of genus \\( g \\), and let \\( L \\) be a holomorphic line bundle on \\( X \\). Let \\( h^0(L) \\) denote the number of linearly independent global sections of \\( L \\).\n\nRiemann-Roch Theorem\n-------------------\n\nFor any holomorphic line bundle \\( L \\) on \\( X \\):\n\n\\[ h^0(L) - h^1(L) = \\deg(L) - g + 1 \\]\n\nwhere \\( \\deg(L) \\) denotes the degree of the line bundle, which is defined as the sum of the degrees of the local trivializations of \\( L \\).\n\nThis formula has far-reaching implications in many areas of mathematics and physics, including string theory, algebraic K-theory, and the study of moduli spaces.\n\nKey Concepts\n-------------\n\n*   **Genus**: A topological invariant of a Riemann surface.\n*   **Line Bundle**: A holomorphic vector bundle of rank one.\n*   **Degree**: An integer-valued function on the set of line bundles on a Riemann surface.\n*   **Global Section**: A section of a line bundle over all points of a Riemann surface.\n*   **Dimension of Global Sections**: The maximum number of linearly independent global sections of a line bundle.\n\nProof Sketch\n-------------\n\nA sketch of the proof involves using the following steps:\n\n1.  Constructing a resolution of singularities for the sheaf associated to the line bundle.\n2.  Applying the Serre duality principle to relate the dimensions of the cohomology groups of the sheaves involved.\n3.  Using the properties of the Euler characteristic to derive the desired formula.\n\nNote that this is just a brief overview of the main ideas behind the Riemann-Roch theorem. For a more detailed treatment, consult advanced texts or research papers on algebraic geometry.",
        "output_token_count": 446,
        "input_token_count": 126
    }
}
```

## Streaming
To do streaming, you can call the endpoint with a similar post request like so:
```python
import requests

response = requests.get(f"http://localhost:8000/api/llm", json={
	"auth": {"api_key": "sk-123456789"},
	"stream": True,
	"question": "What is the Riemann-Roch theorem?",
	"model_parameters": {
		# "model": "llama-3.1-8b-instruct",
		"stream": True, 
		"max_tokens": 1000, 
		"temperature": 0.5, 
		"top_p": 0.9, 
		"repetition_penalty": 1.15
	}
}, stream=True)

response.raise_for_status()

for chunk_raw in response.iter_content(chunk_size=None, decode_unicode=False):
	chunk_decoded = chunk_raw.decode("utf-8")
	print(chunk_decoded, end="")
```
