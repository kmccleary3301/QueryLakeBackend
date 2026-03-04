# fetch_toolchain_sessions API Documentation

### ✅ `fetch_toolchain_sessions`

**Endpoint:** `/api/fetch_toolchain_sessions`  
**Method:** `GET` / `POST`  
**Description:** Get previous toolchain sessions of the user. Returned as a list of objects sorted by timestamp. An optional cutoff date can be provided in unix time.

#### Arguments:
- **auth** (required): Authentication method. Choose one of the following:
  - OAuth2: `{"auth": "oauth2_string"}`
  - API Key: `{"auth": {"api_key": "example_api_key"}}`
  - Username + Password: `{"auth": {"username": "example_username", "password": "example_password"}}`
- **cutoff_date** (optional): A float representing the cutoff date in unix time.

#### Example Usage

**Python Example:**
```python
import requests
import json

input_data = {
    "auth": {"api_key": "example_api_key"},
    "cutoff_date": None  # Optional, can be omitted if not needed
}

response = requests.get("http://localhost:8000/api/fetch_toolchain_sessions", json=input_data)
result = response.json()

print(json.dumps(result, indent=4))

# Ensure there was no error
assert not ("success" in result and result["success"] == False), result["error"]
```

**JavaScript Example:**
```javascript
const fetch = require('node-fetch');

const inputData = {
    auth: { api_key: "example_api_key" },
    cutoff_date: null  // Optional, can be omitted if not needed
};

fetch("http://localhost:8000/api/fetch_toolchain_sessions", {
    method: "GET",
    body: JSON.stringify(inputData),
    headers: { 'Content-Type': 'application/json' }
})
.then(response => response.json())
.then(result => {
    console.log(JSON.stringify(result, null, 4));

    // Ensure there was no error
    if ("success" in result && result.success === false) {
        throw new Error(result.error);
    }
})
.catch(error => console.error('Error:', error));
```

### Response Structure
The response will be structured as follows:
```json
{
    "success": true,
    "result": []  // Array of toolchain session objects
}
```

In case of an error:
```json
{
    "success": false,
    "error": "Error message."
}
``` 

Use this API to effectively retrieve and manage toolchain sessions for users. Ensure appropriate authentication is provided based on your use case.