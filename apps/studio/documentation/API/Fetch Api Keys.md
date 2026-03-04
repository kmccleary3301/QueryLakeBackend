# Fetch API Keys

### ✅ `fetch_api_keys`

The `fetch_api_keys` endpoint allows users to retrieve all API keys associated with their account. This can be done using one of three authentication methods: API Key, OAuth2, or username/password.

### Endpoint
```
GET /api/fetch_api_keys
```

### Request Body
The body of the request should include the following structure, based on the authentication method used. Below is the required field:

- `auth`: Authentication object which can take one of the following forms:

  - For OAuth2:
    ```json
    {"auth": "oauth2_string"}
    ```

  - For API Key:
    ```json
    {"auth": {"api_key": "example_api_key"}}
    ```

  - For Username and Password:
    ```json
    {"auth": {"username": "example_username", "password": "example_password"}}
    ```

### Response Structure
The response will be a JSON object containing:

- `success`: Boolean indicating if the request was successful.
- `result`: Contains the list of API keys, which includes:
  - `api_keys`: An array of API key objects with the following fields:
    - `id`: Unique identifier for the API key.
    - `title`: Name/description of the API key.
    - `created`: Timestamp of when the API key was created.
    - `last_used`: Timestamp of the last time the API key was used (null if never used).
    - `key_preview`: Partial preview of the API key.

### Example Usage

#### Python
```python
import requests, json
from copy import deepcopy

input = deepcopy({"auth": {"api_key": "example_api_key"}})

response = requests.get("http://localhost:8000/api/fetch_api_keys", json=input)
result = response.json()

print(json.dumps(result, indent=4))
```

#### JavaScript
```javascript
const fetch = require('node-fetch');

const input = {"auth": {"api_key": "example_api_key"}};

fetch('http://localhost:8000/api/fetch_api_keys', {
    method: 'GET',
    body: JSON.stringify(input),
    headers: {'Content-Type': 'application/json'}
})
.then(response => response.json())
.then(result => console.log(JSON.stringify(result, null, 4)));
```

This endpoint is particularly useful for managing API keys directly from your application and ensuring that you have access to the necessary keys for making further API requests.