# API Documentation: Embedding Endpoint

### ✅ `embedding`

Call a text to vector embedding model like BGE-M3.

---

#### Endpoint
- **URL**: `http://localhost:8000/api/embedding`
- **Method**: GET / POST

#### Authentication
The endpoint supports the following authentication methods:
- API Key (recommended)
  
```json
{"auth": {"api_key": "example_api_key"}}
```

#### Request Structure
The request should follow the structure below:

```json
{
    "auth": {"api_key": "example_api_key"},
    "inputs": [
        "Your first prompt here",
        "Your second prompt here"
    ]
}
```

#### Example Request (Python)

```python
import requests

API_KEY_1 = "example_api_key"
prompt_input = ["What is the square root of 169?", "What is the derivative of sin(cos(x))?"]

input_data = {
    "auth": {"api_key": API_KEY_1},
    "inputs": prompt_input
}

response = requests.get("http://localhost:8000/api/embedding", json=input_data)
response_value = response.json()
print(response_value)
```

#### Example Request (JavaScript)

```javascript
const fetch = require("node-fetch");

const API_KEY_1 = "example_api_key";
const promptInput = [
    "What is the square root of 169?",
    "What is the derivative of sin(cos(x))?"
];

const inputData = {
    auth: { api_key: API_KEY_1 },
    inputs: promptInput
};

fetch("http://localhost:8000/api/embedding", {
    method: "GET",
    body: JSON.stringify(inputData),
    headers: { "Content-Type": "application/json" }
})
.then(response => response.json())
.then(responseValue => console.log(responseValue));
```

#### Response Structure
A successful response will look like this:

```json
{ 
	"success": true,
	"result": [
		// Example embedding values
		[1, 2, 3, ...],
		[1, 2, 3, ...]
	]
}
```


### Notes
- The `embedding` endpoint can process multiple prompts. You can pass an array of prompts under the "inputs" field.
- Ensure that you replace `example_api_key` with your actual API key when making requests.

### Result
