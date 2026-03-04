# Craft Document Access Token API Documentation

### ✅ `craft_document_access_token`

This API endpoint allows users to craft a document access token using the global server public key. By default, the token expires in 60 seconds, but clients can specify a different validity window.

#### Endpoint
```
POST /api/craft_document_access_token
GET /api/craft_document_access_token
```

#### Authentication
This endpoint can be accessed using any of the following authentication methods (only one method is allowed per request):
- **API Key**:
  ```json
  {"auth": {"api_key": "example_api_key"}}
  ```
- **OAuth2**:
  ```json
  {"auth": "oauth2_string"}
  ```
- **Username and Password**:
  ```json
  {"auth": {"username": "example_username", "password": "example_password"}}
  ```

#### Request Body
The request must include the following parameters in JSON format:
- `auth`: Authentication method (as described above)
- `hash_id`: (string) A unique identifier for the document.
- `validity_window`: (float, optional) Time in seconds until the token expires. Default is 60 seconds.

#### Response
On a successful call, the response will be in the following format:
```json
{
    "success": true,
    "result": {
        "file_name": "example_file_name.pdf",
        "access_encrypted": "encrypted_access_token_here"
    }
}
```
In the case of an error, a response will look like this:
```json
{
    "success": false,
    "error": "error_message_here"
}
```

#### Example Usage

**Python Example:**
```python
import requests, json

input = {
    "auth": {"api_key": "example_api_key"},
    "hash_id": "example_hash_id",
    "validity_window": 60
}

response = requests.get("http://localhost:8000/api/craft_document_access_token", json=input)
result = response.json()

print(json.dumps(result, indent=4))
```

**JavaScript Example:**
```javascript
const axios = require('axios');

const input = {
    auth: { api_key: "example_api_key" },
    hash_id: "example_hash_id",
    validity_window: 60
};

axios.get("http://localhost:8000/api/craft_document_access_token", { data: input })
    .then(response => {
        console.log(JSON.stringify(response.data, null, 4));
    });
```

In both examples, replace `"example_hash_id"` and `"example_api_key"` with appropriate values. The API will return the crafted document access token if the request is successful.

# API Documentation for `fetch_document`

### ✅ `fetch_document`

Returns a link to open or download a document. Meant to be used with `craft_document_access_token`.

### Endpoint
```
GET /api/fetch_document
```

### Authentication
This endpoint supports the following authentication methods for access:

- **API Key**: 
    ```json
    {"auth": {"api_key": "example_api_key"}}
    ```

- **Username and Password**: 
    ```json
    {"auth": {"username": "example_username", "password": "example_password"}}
    ```

- **OAuth2**: 
    ```json
    {"auth": "oauth2_string"}
    ```

**Note:** The `fetch_document` API can be called using any of the supported authentication methods, but other endpoints like `create_api_key` and `delete_api_key` require OAuth2 or username/password.

### Function Arguments
- **document_auth_access** (str): The authentication access token for the document.

### Example Requests

#### Python Example

```python
import requests
import json
from urllib.parse import urlencode
from copy import deepcopy

# Replace this with your document authentication access token
DOCUMENT_AUTH_ACCESS = {"document_auth_access": "your_access_token"}

input = deepcopy(DOCUMENT_AUTH_ACCESS)
encoded_params = urlencode({"parameters": json.dumps(input)})

response = requests.get("http://localhost:8000/api/fetch_document?" + encoded_params, 
                        json={"auth": {"api_key": "example_api_key"}})
response.raise_for_status()

result = response.json()
print(json.dumps(result, indent=4))
```

#### JavaScript Example

```javascript
const axios = require('axios');

let documentAuthAccess = { document_auth_access: "your_access_token" };

let params = new URLSearchParams({ parameters: JSON.stringify(documentAuthAccess) });

axios.get(`http://localhost:8000/api/fetch_document?${params}`, {
    data: { auth: { api_key: "example_api_key" } }
})
.then(response => {
    console.log(JSON.stringify(response.data, null, 4));
})
.catch(error => {
    console.error(error);
});
```

### Response
The response will contain a link to the document. This endpoint allows users to securely fetch documents in a streaming manner suitable for viewing or downloading. Always ensure the correct authentication method is used for best results.