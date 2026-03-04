# Fetch All Collections API Documentation

The `fetch_all_collections` endpoint allows you to retrieve all collections that a user has the privilege to read. This endpoint supports various authentication methods, allowing for flexibility in API access.

## Endpoint

```
GET /api/fetch_all_collections
POST /api/fetch_all_collections
```

### Authentication

The following authentication methods are supported:

- **API Key**
    ```json
    {"auth": {"api_key": "example_api_key"}}
    ```
  
- **Username and Password** (for login)
    ```json
    {"auth": {"username": "example_username", "password": "example_password"}}
    ```

- **OAuth2** (for protected operations)
    ```json
    {"auth": "oauth2_string"}
    ```

### Description

Fetches all collections that a user has the privilege to read. 

### Arguments

- `auth` (required): The authentication method used to access the API.

### Example Requests

#### Python Example

```python
import requests, json

# Sample input with API key authentication
input = {
    "auth": {"api_key": "example_api_key"}
}
response = requests.get("http://localhost:8000/api/fetch_all_collections", json=input)
result = response.json()

print(json.dumps(result, indent=4))
```

#### JavaScript Example

```javascript
fetch("http://localhost:8000/api/fetch_all_collections", {
    method: "GET",
    body: JSON.stringify({
        auth: { api_key: "example_api_key" }
    }),
    headers: {
        "Content-Type": "application/json"
    }
})
.then(response => response.json())
.then(result => console.log(JSON.stringify(result, null, 4)));
```

### Example Response

A successful response will have the following structure:

```json
{
    "success": true,
    "result": {
        "collections": {
            "global_collections": [],
            "user_collections": [
                {
                    "name": "rag_collection",
                    "hash_id": "FmBtLFCl0ZTwM98imoDtAvRfqZIKycXe",
                    "document_count": 0,
                    "type": "user"
                }
            ],
            "organization_collections": []
        }
    }
}
```

### Notes

- The endpoint supports both `GET` and `POST` methods.
- Ensure that you utilize one of the supported authentication methods to access the endpoint successfully.
- The `add_user` function does not require authentication but may be restricted based on server configuration.