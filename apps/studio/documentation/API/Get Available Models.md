# API Documentation for `get_available_models`

### ✅ `get_available_models`

#### Description
The `get_available_models` endpoint retrieves a list of all models available on the server for the authenticated user. Future implementations may enable organizations to have private models.

#### Endpoint
```
GET /api/get_available_models
```

#### Authentication
This endpoint supports three types of authentication:
1. OAuth2:
   ```json
   {"auth": "oauth2_string"}
   ```

2. API Key:
   ```json
   {"auth": {"api_key": "example_api_key"}}
   ```

3. Username and Password:
   ```json
   {"auth": {"username": "example_username", "password": "example_password"}}
   ```

It is important to note that **only one** authentication method is allowed per request.

#### Request Example

**Python:**
```python
import requests, json

input_data = {
    "auth": {"api_key": "example_api_key"}
}

response = requests.get("http://localhost:8000/api/get_available_models", json=input_data)
result = response.json()

print(json.dumps(result, indent=4))
```

**JavaScript:**
```javascript
fetch("http://localhost:8000/api/get_available_models", {
    method: "GET",
    body: JSON.stringify({
        auth: { api_key: "example_api_key" }
    }),
    headers: {
        "Content-Type": "application/json"
    }
})
.then(response => response.json())
.then(result => console.log(JSON.stringify(result, null, 4)));
```

#### Response Structure
Upon a successful request, the JSON response will include an object with the following structure:

```json
{
    "success": true,
    "result": {
        "available_models": {
            "default_models": {
                "llm": "llama-3.1-8b-instruct",
                "rerank": "bge-reranker-v2-m3",
                "embedding": "bge-m3"
            },
            "local_models": [
                {
                    "name": "Qwen2 VL 7B Instruct (AWQ)",
                    "id": "qwen2-vl-7b-instruct",
                    "modelcard": "https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct-AWQ"
                },
                {
                    "name": "LLaMA 3.1 8B Instruct (AQLM PV 2BPW)",
                    "id": "llama-3.1-8b-instruct",
                    "modelcard": "https://ai.meta.com/blog/meta-llama-3-1/"
                },
                {
                    "name": "LLaMA 3.1 70B Instruct (AQLM PV 2BPW)",
                    "id": "llama-3.1-70b-instruct",
                    "modelcard": "https://ai.meta.com/blog/meta-llama-3-1/"
                }
            ],
            "external_models": {
                "openai": [
                    {
                        "name": "GPT-4 Turbo"
                    }
                ]
            }
        }
    }
}
```

#### Note
This endpoint supports both GET and POST requests, although typically the GET method is used for retrieving data.