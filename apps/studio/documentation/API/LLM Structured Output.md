# API Documentation for `llm` Endpoint

### ✅ `llm` [Structured Output]

Call an LLM with TypeScript/Pydantic scheme enforcement on the response.

#### Endpoint
`GET` or `POST http://localhost:8000/api/llm`

#### Authentication
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

### Request Body

- **auth**: The authentication method (as shown above).
- **model_parameters**: An object containing parameters specific to the model.
  - **grammar**: A list containing the defined grammar for the model (TypeScript schema).
- **question**: The question to send to the LLM.

### Example Usage

#### Python Example

```python
import requests
import json

API_KEY_1 = "example_api_key"  # Substitute with your actual API key
ts_scheme = """
type User = {
    name: string;
    age: number;
    isActive: boolean;
};
"""

response = requests.get(f"http://localhost:8000/api/llm", json={
    "auth": {"api_key": API_KEY_1},
    "model_parameters": {
        "grammar": ["typescript", ts_scheme]
    },
    "question": "Give me a current NBA player. Respond as a JSON.",
})

result = response.json()

print(json.dumps(result, indent=4))
if "trace" in result:
    print(result["trace"])
```

#### JavaScript Example

```javascript
const fetch = require('node-fetch');

const API_KEY_1 = "example_api_key";  // Substitute with your actual API key
const ts_scheme = `
type User = {
    name: string;
    age: number;
    isActive: boolean;
};
`;

fetch("http://localhost:8000/api/llm", {
    method: "GET",
    body: JSON.stringify({
        auth: { api_key: API_KEY_1 },
        model_parameters: {
            grammar: ["typescript", ts_scheme]
        },
        question: "Give me a current NBA player. Respond as a JSON."
    }),
    headers: {
        'Content-Type': 'application/json'
    }
})
.then(response => response.json())
.then(result => {
    console.log(JSON.stringify(result, null, 4));
    if (result.trace) {
        console.log(result.trace);
    }
})
.catch(error => console.error("Error:", error));
```

### Sample Output
```json
{
    "success": true,
    "result": {
        "output": "{\n  \"name\": \"Luka Doncic\",\n  \"age\": 24,\n  \"isActive\": true\n}\n}",
        "output_token_count": 32,
        "input_token_count": 129
    }
}
```

Please note that it is technically possible for it to return only whitespaces as a response, so we recommend explaining the scheme in your prompt. 