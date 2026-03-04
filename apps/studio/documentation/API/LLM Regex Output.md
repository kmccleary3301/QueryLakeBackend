### ✅ `llm` [Regex Output]

Call an LLM with the response being forced to a provided regex.

#### Endpoint

```
GET /api/llm
POST /api/llm
```

#### Authentication

This endpoint supports the following authentication method:

- **API Key**: 
  ```json
  {"auth": {"api_key": "example_api_key"}}
  ```

#### Request Parameters

- **auth**: Authentication object as described above.
- **model_parameters**: 
  - **grammar**: An array where the first element is the type ("regex") and the second element is the regex pattern to enforce on the output.
- **question**: The question to ask the LLM.

#### Request Examples

**Python Example:**

```python
import requests
import json

API_KEY = "example_api_key"
regex_scheme = r'(0?[1-9]|1[0-2])\/(0?[1-9]|1\d|2\d|3[01])\/(19|20)\d{2}'

response = requests.get("http://localhost:8000/api/llm", json={
    "auth": {"api_key": API_KEY},
    "model_parameters": {
        "grammar": ["regex", regex_scheme],
    },
    "question": "Who is Oda? Write it as a JSON.",
})

result = response.json()
print(json.dumps(result, indent=4))
```

**JavaScript Example:**

```javascript
const API_KEY = "example_api_key";
const regex_scheme = '(0?[1-9]|1[0-2])/(0?[1-9]|1\\d|2\\d|3[01])/(19|20)\\d{2}';

fetch("http://localhost:8000/api/llm", {
    method: "GET",
    body: JSON.stringify({
        auth: { api_key: API_KEY },
        model_parameters: {
            grammar: ["regex", regex_scheme],
        },
        question: "Who is Oda? Write it as a JSON.",
    }),
    headers: {
        "Content-Type": "application/json",
    },
})
.then(response => response.json())
.then(result => console.log(JSON.stringify(result, null, 4)))
.catch(error => console.error('Error:', error));
```

#### Response Structure

The response will contain the following structure:

```json
{
    "success": true,
    "result": {
        "output": "1/2/2023",
        "output_token_count": 7,
        "input_token_count": 128
    }
}
```

- **success**: Boolean indicating if the request was successful.
- **result**: Object containing:
  - **output**: The generated output based on the provided question and regex.
  - **output_token_count**: The number of tokens in the output.
  - **input_token_count**: The number of tokens in the input question.