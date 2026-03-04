# API Documentation: `delete_document`

### ✅ `delete_document`

The `delete_document` endpoint authorizes that the user has permission to delete a document, and then performs the deletion. This endpoint supports both GET and POST requests and requires authentication to access.

**Endpoint**: `/api/delete_document`  
**Method**: GET, POST

### Authentication
This endpoint supports three types of authentication:
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

### Parameters
| Keyword   | Type                               | Default  | Description                                          |
|-----------|-------------------------------------|----------|------------------------------------------------------|
| `auth`    | `QUERYLAKE_AUTH`                  | N/A      | Authentication credentials (one of the three types) |
| `hash_id` | `Union[List[str], str]`           | `None`   | The unique identifier for the document to delete     |
| `hash_ids`| `Union[List[str], str]`           | `None`   | List of unique identifiers for multiple documents to delete |

### Example Requests

#### Python Example

```python
import requests, json

input = {
    "auth": {"api_key": "example_api_key"},
    "hash_id": "unique_document_id"
}

response = requests.get("http://localhost:8000/api/delete_document", json=input)
result = response.json()

print(json.dumps(result, indent=4))
```

#### JavaScript Example

```javascript
const fetch = require('node-fetch');

const input = {
    auth: { api_key: "example_api_key" },
    hash_id: "unique_document_id"
};

fetch("http://localhost:8000/api/delete_document", {
    method: "GET",
    body: JSON.stringify(input),
    headers: {
        'Content-Type': 'application/json'
    }
})
.then(response => response.json())
.then(result => console.log(JSON.stringify(result, null, 4)));
```

### Response Structure
On success, the response would be structured as follows:

```json
{
    "success": true
}
```

### Notes
- The API key authentication method is suitable for simple applications needing basic access controls. If your application requires user-specific actions, consider using OAuth2 or username/password authentication methods.
- Keep in mind that the `hash_id` parameter is required to define which document you want to delete. You can also specify multiple documents using the `hash_ids` parameter, but not both at the same time.