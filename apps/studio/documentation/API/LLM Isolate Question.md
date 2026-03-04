# API Documentation: `llm_isolate_question`

### ✅ `llm_isolate_question`

This endpoint extracts and isolates the most recent question from a chat history, making it clearer without requiring additional context. It is particularly useful for Retrieval-Augmented Generation (RAG) tasks.

## Endpoint

- **URL:** `http://localhost:8000/api/llm_isolate_question`
- **Methods:** GET, POST

## Authentication

This endpoint supports authentication through the following methods:

1. **API Key**:
    ```json
    {"auth": {"api_key": "example_api_key"}}
    ```

2. **Username & Password**:
    ```json
    {"auth": {"username": "example_username", "password": "example_password"}}
    ```

3. **OAuth2** (not applicable for this endpoint unless creating/deleting an API key).

## Parameters

- **auth**: Authentication object (see above).
- **chat_history**: A list of dictionaries representing the chat history. Each dictionary should contain the role (`user` or `assistant`) and the content of the message.
- **model_choice** (optional): A string to specify the model choice, defaulting to `None`.

## Example Request

### Python
```python
import requests

# Example chat history
chat_history = [
    {"role": "user", "content": "Which theorem allows us to measure the extent to which the fundamental theorem of calculus fails at high dimensional manifolds?"},
    {"role": "assistant", "content": "The theorem you're referring to ..."},
    {"role": "user", "content": "Tell me more about FTC."}
]

response = requests.post("http://localhost:8000/api/llm_isolate_question", json={
    "auth": {"api_key": "example_api_key"},
    "chat_history": chat_history
})

result = response.json()
print(result)
```

### JavaScript
```javascript
const fetch = require('node-fetch');

const chatHistory = [
    { role: "user", content: "Which theorem allows us to measure the extent to which the fundamental theorem of calculus fails at high dimensional manifolds?" },
    { role: "assistant", content: "The theorem you're referring to ..." },
    { role: "user", content: "Tell me more about FTC." }
];

fetch("http://localhost:8000/api/llm_isolate_question", {
    method: "POST",
    body: JSON.stringify({
        auth: { api_key: "example_api_key" },
        chat_history: chatHistory
    }),
    headers: {
        "Content-Type": "application/json"
    }
})
.then(response => response.json())
.then(data => console.log(data));
```

## Response Structure

The response will be a JSON object indicating the success of the operation and any results or errors.

### Sample Output
```json
{
    "success": true,
    "result": false
}
```

## Notes

- This endpoint does not require user authentication for adding a user but it may be disabled based on server configuration.
- Make sure to check if your authentication type aligns with the operation you're performing, especially when creating or deleting API keys, which may require OAuth2 or username/password authentication.