# API Documentation: `get_available_toolchains`

### ✅ `get_available_toolchains`

**Endpoint**: `/api/get_available_toolchains`  
**Method**: GET, POST  
**Description**: This endpoint returns the available toolchains, including chat window settings. If there are organization-locked toolchains, they will also be added to the database.

**Authentication Options**: The endpoint supports the following authentication methods:
- **API Key**:
  ```json
  {"auth": {"api_key": "example_api_key"}}
  ```
- **Username + Password**:
  ```json
  {"auth": {"username": "example_username", "password": "example_password"}}
  ```
- **OAuth2**:
  ```json
  {"auth": "oauth2_string"}
  ```

**Function Arguments**:
- `auth` (required): Authentication information based on the method chosen.

## Example Usage

### Python Example
```python
import requests
import json

# Replace with your API key
input_data = {
    "auth": {"api_key": "example_api_key"}
}

response = requests.get("http://localhost:8000/api/get_available_toolchains", json=input_data)
result = response.json()

print(json.dumps(result, indent=4))
```

### JavaScript Example
```javascript
fetch("http://localhost:8000/api/get_available_toolchains", {
    method: "GET",
    body: JSON.stringify({
        auth: { api_key: "example_api_key" }
    }),
    headers: {
        "Content-Type": "application/json"
    }
})
.then(response => response.json())
.then(result => {
    console.log(JSON.stringify(result, null, 4));
});
```

## Response Structure
A successful response will contain the following structure:
```json
{
    "success": true,
    "result": {
        "toolchains": [
            {
                "category": "Test",
                "entries": [
                    {
                        "title": "Iteration Test",
                        "id": "iterable_test",
                        "category": "Test"
                    },
                    {
                        "title": "File Upload OCR Test",
                        "id": "test_file_upload",
                        "category": "Test"
                    },
                    {
                        "title": "Static Chat Test And Really Long Name",
                        "id": "test_chat_session_normal",
                        "category": "Test"
                    },
                    {
                        "title": "Streaming Chat Test",
                        "id": "test_chat_session_normal_streaming",
                        "category": "Test"
                    }
                ]
            }
        ]
    }
}
```

### Error Handling
While the provided examples do not include error handling, it is advisable to check the `success` field in the response for error detection, particularly if the value is `false`.