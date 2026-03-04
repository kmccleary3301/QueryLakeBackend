# API Documentation - `delete_api_key`

### ✅ `delete_api_key`

**Description:**  
Delete an API key by its id.

**Endpoint:**  
`/api/delete_api_key`

**Supported Methods:**  
- GET
- POST

**Authentication Methods:**  
This endpoint supports authentication in one of the following ways (only one type per request):
1. API Key
   ```json
   {"auth": {"api_key": "example_api_key"}}
   ```
2. OAuth2
   ```json
   {"auth": "oauth2_string"}
   ```
3. Username and Password
   ```json
   {"auth": {"username": "example_username", "password": "example_password"}}
   ```

**Required Parameters:**
- `auth`: Required authentication method (as detailed above).
- `api_key_id`: The ID of the API key you wish to delete (type: string).

**Response:**  
A successful response will return a JSON object indicating success:
```json
{
    "success": true
}
```

---

### Python Example

```python
import requests, json
from copy import deepcopy

# Prepare the request data
input = deepcopy({"auth": {"username": "example_username", "password": "example_password"}})
input.update({"api_key_id": "example_api_key_id"})

# Send the request
response = requests.get("http://localhost:8000/api/delete_api_key", json=input)
result = response.json()

# Print and process the result
print(json.dumps(result, indent=4))
```

---

### JavaScript Example

```javascript
const axios = require('axios');

// Prepare the request data
const input = {
    auth: {
        username: "example_username",
        password: "example_password"
    },
    api_key_id: "example_api_key_id"
};

// Send the request
axios.get('http://localhost:8000/api/delete_api_key', { data: input })
    .then(response => {
        console.log(response.data);
    })
    .catch(error => {
        console.error(error);
    });
```

---

This API endpoint allows you to securely delete an API key from your account by providing the key's ID and authenticating your request appropriately.