# API Documentation: `function_help`

### ✅ `function_help`

Get the specs of all available API endpoints.

---

### Request Structure

To access the `function_help` endpoint, you must send a POST request with a JSON body that includes authentication details. You can authenticate using an API key, OAuth2 string, or by providing a username and password. However, for this request, we will focus on the API key authentication method.

- **Endpoint:** `POST /api/function_help`
- **Required Authentication:** User's API key

#### Example Request JSON (API Key)

```json
{
    "auth": {
        "api_key": "example_api_key"
    }
}
```

### Example Code Snippets

#### Python Example

```python
import requests
import json

input = {"auth": {"api_key": "example_api_key"}}

response = requests.post("http://localhost:8000/api/function_help", json=input)
response.raise_for_status()

result = response.json()

print(json.dumps(result, indent=4))
assert not ("success" in result and result["success"] == False), result["error"]

if "result" in result:
    print(json.dumps([r["function_name"] for r in result["result"]], indent=4))
```

#### JavaScript Example

```javascript
const axios = require('axios');

const input = {
    auth: {
        api_key: "example_api_key"
    }
};

axios.post("http://localhost:8000/api/function_help", input)
    .then(response => {
        console.log(JSON.stringify(response.data, null, 4));
        if (response.data.result) {
            console.log(JSON.stringify(response.data.result.map(r => r.function_name), null, 4));
        }
    })
    .catch(error => {
        console.error(error);
    });
```

### Response Structure

On a successful request, the response will include a JSON object containing an array of available API functions.

#### Example Response

```json
{
    "success": true,
    "result": [
        {
            "endpoint": "/api/retrieve_toolchain_session_from_db",
            "api_function_id": "retrieve_toolchain_session_from_db",
            "function_name": "retrieve_toolchain_session_from_db",
            "description": "",
            "function_args": [
                {
                    "keyword": "auth",
                    "type_hint": "QUERYLAKE_AUTH"
                },
                {
                    "keyword": "session_id",
                    "type_hint": "str"
                },
                {
                    "keyword": "ws",
                    "type_hint": "starlette.websockets.WebSocket"
                }
            ]
        },
        {
            "endpoint": "/api/parse_PDFs",
            "api_function_id": "parse_PDFs",
            "function_name": "parse_PDFs",
            "description": "",
            "function_args": [
                // Additional arguments here
            ]
        }
    ]
}
```

### Additional Notes

- Each API function listed under `"result"` contains details such as the endpoint, function ID, name, a description, and the required arguments along with their type hints.
- Ensure you handle error checks in production code, especially for response validation.