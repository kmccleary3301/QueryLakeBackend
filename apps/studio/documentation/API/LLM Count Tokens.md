# API Documentation: `llm_count_tokens`

### ✅ `llm_count_tokens`

Count the tokens for a given string using the tokenizer of a specified model.

---

## Endpoint

- **URL**: `http://localhost:8000/api/llm_count_tokens`
- **Method**: `GET` or `POST`

## Authentication

This endpoint supports the following authentication methods:

1. **API Key** (recommended for general use):
   ```json
   {
       "auth": {"api_key": "example_api_key"}
   }
   ```

2. **Username and Password** (used for creating API keys):
   ```json
   {
       "auth": {"username": "example_username", "password": "example_password"}
   }
   ```

3. **OAuth2** (used for creating API keys):
   ```json
   {
       "auth": "oauth2_string"
   }
   ```

## Request Parameters

### Required Payload

- **model_id** (str): The ID of the model for tokenization.
- **input_string** (str): The string to count tokens for.

### Example Request Body

```json
{
    "auth": {"api_key": "example_api_key"},
    "model_id": "llama-3.1-8b-instruct",
    "input_string": "Where is Afghanistan?"
}
```

---

## Example Code Snippets

### Python

```python
import requests

response = requests.get("http://localhost:8000/api/llm_count_tokens", json={
    "auth": {"api_key": "example_api_key"},
    "model_id": "llama-3.1-8b-instruct",
    "input_string": "Where is Afghanistan?"
})

result = response.json()
print(result)
```

### JavaScript

```javascript
fetch("http://localhost:8000/api/llm_count_tokens", {
    method: "GET",
    body: JSON.stringify({
        auth: { api_key: "example_api_key" },
        model_id: "llama-3.1-8b-instruct",
        input_string: "Where is Afghanistan?"
    }),
    headers: {
        "Content-Type": "application/json",
    },
})
.then(response => response.json())
.then(result => console.log(result));
```

--- 

Feel free to utilize this endpoint to facilitate token counting based on the specified model and input string.