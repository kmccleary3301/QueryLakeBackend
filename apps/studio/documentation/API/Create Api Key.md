# Create API Key

## ✅ `create_api_key`

Create a new API key for the user. This endpoint requires authentication and accepts a title for the new API key.

### Endpoint
```
POST /api/create_api_key
```

### Authentication
The `create_api_key` endpoint supports the following authentication methods:
- **OAuth2**
```json
{"auth": "oauth2_string"}
```
- **API Key** (not supported for this endpoint)
- **Username + Password**
```json
{"auth": {"username": "example_username", "password": "example_password"}}
```

### Parameters
- `title` (optional): A string that represents the title of the API key.

### Request Example

Here’s how to call the `create_api_key` endpoint using both Python and JavaScript.

#### Python Example
```python
import requests
import json

def create_api_key(title: str):
    input = {"auth": {"username": "example_username", "password": "example_password"}}
    input.update({"title": title})

    response = requests.post("http://localhost:8000/api/create_api_key", json=input)
    result = response.json()

    print(json.dumps(result, indent=4))
    return result["result"]

API_KEY_INFO = create_api_key("API Key 1")
API_KEY = API_KEY_INFO["api_key"]

print("API_KEY:", API_KEY)
```

#### JavaScript Example
```javascript
async function createApiKey(title) {
    const input = {
        auth: { username: "example_username", password: "example_password" },
        title: title
    };

    const response = await fetch("http://localhost:8000/api/create_api_key", {
        method: "POST",
        body: JSON.stringify(input),
        headers: { 'Content-Type': 'application/json' }
    });

    const result = await response.json();
    console.log(JSON.stringify(result, null, 4));
    return result.result;
}

createApiKey("API Key 1").then(apiKeyInfo => {
    console.log("API_KEY:", apiKeyInfo.api_key);
});
```

### Response Example
Upon success, the response will contain a structure like this:
```json
{
    "success": true,
    "result": {
        "api_key": "sk-6kwHVPIcgnRtz3LE2WixpqXkvbPEyDRt4QEukkK4vZF4JfaN",
        "id": "DY5Rq6HFTt8tsUfgSMNNpvfoY0ez6M0W",
        "title": "API Key 1",
        "created": 1728425378.0665762,
        "last_used": null,
        "key_preview": "sk-...JfaN"
    }
}
```

### Notes
- The API key is created only with valid username and password credentials. 
- The success response indicates that the key has been successfully generated, along with its details.