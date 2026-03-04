# API Documentation for `llm_make_conversation_title`

### ✅ `llm_make_conversation_title`

Generate a conversation title from chat history using an LLM. This function is useful for creating chat labels for the sidebar.

#### Endpoint
`GET /api/llm_make_conversation_title`  
`POST /api/llm_make_conversation_title`

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

#### Request Payload
The request JSON must contain the following fields:
- `auth`: Authentication method as described above.
- `chat_history`: A list of dictionaries representing the chat history.

Optional parameters can also be included:
- `stream_callables`: A dictionary of callable functions for streaming responses. Default is `None`.
- `model_choice`: A string specifying the model choice. Default is `None`.

#### Example Request

**Python Example:**
```python
import requests

API_KEY_1 = "example_api_key"
TEST_CHAT_HISTORY = [{"text": "Hello! How are you?"}, {"text": "What's the weather today?"}]

response = requests.get("http://localhost:8000/api/llm_make_conversation_title", json={
    "auth": {"api_key": API_KEY_1},
    "chat_history": TEST_CHAT_HISTORY
})
result = response.json()

print(result)
```

**JavaScript Example:**
```javascript
const axios = require('axios');

const API_KEY_1 = "example_api_key";
const TEST_CHAT_HISTORY = [{ text: "Hello! How are you?" }, { text: "What's the weather today?" }];

axios.get("http://localhost:8000/api/llm_make_conversation_title", {
    auth: { api_key: API_KEY_1 },
    data: {
        chat_history: TEST_CHAT_HISTORY
    }
}).then(response => {
    console.log(response.data);
}).catch(error => {
    console.error(error);
});
```

#### Response Structure
On success, the API will return a JSON object similar to the following:
```json
{
    "success": true,
    "result": {
        "output": "Mathematical Foundations",
        "output_token_count": 5,
        "input_token_count": 1668
    }
}
```

In case of an error, the response may include additional error details:
```json
{
    "success": false,
    "error": "Error message",
    "trace": "Error trace"
}
```

### Notes
- The `add_user` endpoint does not require any authentication but may be disabled depending on server configuration.
- `create_api_key` and `delete_api_key` actions require OAuth2 or username/password authentication.