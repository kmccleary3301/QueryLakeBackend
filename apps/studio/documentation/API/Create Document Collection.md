# API Documentation for Creating a Document Collection

### ✅ `create_document_collection`

Create a searchable collection of documents, similar to a playlist. If no `hash_id` is given, one is created and returned.

#### Endpoint
```
POST /api/create_document_collection
GET /api/create_document_collection
```

#### Authentication
You can authenticate using one of the following methods:

- **API Key**
    ```json
    {"auth": {"api_key": "example_api_key"}}
    ```
  
- **OAuth2**
    ```json
    {"auth": "oauth2_string"}
    ```

- **Username + Password**
    ```json
    {"auth": {"username": "example_username", "password": "example_password"}}
    ```

#### Request Parameters
| Parameter         | Type          | Description                                               |
|-------------------|---------------|-----------------------------------------------------------|
| `auth`            | QUERYLAKE_AUTH | Authentication method (required)                         |
| `name`            | `str`        | Name of the collection (required)                         |
| `description`     | `str`        | Description of the collection (optional, default: None)  |
| `public`          | `bool`       | Visibility of the collection (optional, default: False)  |
| `organization_id` | `int`        | ID of the organization (optional, default: None)         |

#### Response Structure
A successful request returns a JSON object with the following structure:

```json
{
    "success": true,
    "result": {
        "hash_id": "FmBtLFCl0ZTwM98imoDtAvRfqZIKycXe"
    }
}
```

#### Example Usage

##### Python Example
```python
import requests, json

input = {
    "auth": {"api_key": "example_api_key"},
    "name": "rag_collection",
    "description": "A collection of ragtime music documents",
    "public": False,
    "organization_id": None
}

response = requests.post("http://localhost:8000/api/create_document_collection", json=input)
result = response.json()

print(json.dumps(result, indent=4))
```

##### JavaScript Example
```javascript
const fetch = require('node-fetch');

const input = {
    auth: { api_key: 'example_api_key' },
    name: 'rag_collection',
    description: 'A collection of ragtime music documents',
    public: false,
    organization_id: null
};

fetch('http://localhost:8000/api/create_document_collection', {
    method: 'POST',
    body: JSON.stringify(input),
    headers: { 'Content-Type': 'application/json' }
})
.then(response => response.json())
.then(result => console.log(JSON.stringify(result, null, 4)));
```

This documentation provides a concise overview of how to create a document collection using the API endpoint. Make sure to replace the authentication details with your valid credentials or API keys.