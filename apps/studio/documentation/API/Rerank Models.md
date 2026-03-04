# Rerank API Documentation

### ✅ `rerank`

Call a reranker model to rate the answer relevance of question-answer pairs.

#### Endpoint
- **URL**: `http://localhost:8000/api/rerank`
- **Method**: `GET` or `POST`

#### Authentication
This endpoint supports authentication via API Key, OAuth2, or username and password. Use any one of the following methods:
1. **API Key Authentication**
   ```json
   {"auth": {"api_key": "example_api_key"}}
   ```

2. **OAuth2 Authentication**
   ```json
   {"auth": "oauth2_string"}
   ```

3. **Username and Password Authentication**
   ```json
   {"auth": {"username": "example_username", "password": "example_password"}}
   ```

#### Request Body
The request payload should contain the following:
- **auth**: (required) Authentication method.
- **inputs**: (required) An array of question-answer pairs to be reranked.

##### Example Request Body:
```json
{
    "auth": {"api_key": "example_api_key"},
    "inputs": [
        ["What is the square root of 169?", "The square root of 169 is 13."],
        ["What is the derivative of sin(cos(x))?", "What is the derivative of sin(cos(x))? Well, it's a bit complicated. The derivative of sin(cos(x)) is cos(cos(x)) * -sin(x)."]
    ]
}
```

#### Response Structure
The response will contain:
- `success`: (boolean) Indicates if the request was successful.
- `result`: (array) List of rerank scores for the input question-answer pairs.

##### Example Response:
```json
{
    "success": true,
    "result": [
        1.0827401638380252e-05,
        0.0024664821103215218
    ]
}
```

### Example Code Snippets

#### Python Example
```python
import requests

API_KEY = "example_api_key"

def get_response(prompt_input):
    input_data = {
        "auth": {"api_key": API_KEY},
        "inputs": prompt_input
    }
    
    response = requests.get("http://localhost:8000/api/rerank", json=input_data)
    response_value = response.json()
    
    print(response_value)

prompts_rerank = [
    [
        ["What is the square root of 169?", "The square root of 169 is 13."],
        ["What is the derivative of sin(cos(x))?", "What is the derivative of sin(cos(x))? Well, it's a bit complicated. The derivative of sin(cos(x)) is cos(cos(x)) * -sin(x)."]
    ]
]

for p in prompts_rerank:
    get_response(p)
```

#### JavaScript Example
```javascript
const fetch = require('node-fetch');

const API_KEY = "example_api_key";

const getResponse = async (promptInput) => {
    const inputData = {
        auth: { api_key: API_KEY },
        inputs: promptInput
    };

    const response = await fetch("http://localhost:8000/api/rerank", {
        method: "GET",
        body: JSON.stringify(inputData),
        headers: { 'Content-Type': 'application/json' }
    });
    
    const responseValue = await response.json();
    console.log(responseValue);
};

const promptsRerank = [
    [
        ["What is the square root of 169?", "The square root of 169 is 13."],
        ["What is the derivative of sin(cos(x))?", "What is the derivative of sin(cos(x))? Well, it's a bit complicated. The derivative of sin(cos(x)) is cos(cos(x)) * -sin(x)."]
    ]
];

for (const p of promptsRerank) {
    getResponse(p);
}
```

This documentation outlines the `rerank` API's functionality, usage, and request/response structure to facilitate integration into applications.