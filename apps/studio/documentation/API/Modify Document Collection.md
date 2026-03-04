# Modify Document Collection API

### ✅ `modify_document_collection`

The `modify_document_collection` endpoint allows you to change properties of a document collection for a user. This includes updating the title and description of a specific collection identified by its hash ID. 

**Endpoint:** `/api/modify_document_collection`  
**Method:** GET, POST  
**Authentication:** Supported through API Key, OAuth2, or Username/Password (for creating or deleting API keys only).

#### Authentication
This endpoint supports three types of authentication. Only one is required for any request:
1. **API Key**
   ```json
   {"auth": {"api_key": "example_api_key"}}
   ```
2. **Username and Password**
   ```json
   {"auth": {"username": "example_username", "password": "example_password"}}
   ```
3. **OAuth2 Token**
   ```json
   {"auth": "oauth2_string"}
   ```

#### Function Args

- **auth**: (REQUIRED) Authentication details based on your method of choice.  
- **collection_hash_id**: (REQUIRED) The hash ID of the collection you wish to modify.
- **title**: (OPTIONAL) The title for the document collection.  
- **description**: (OPTIONAL) The description for the document collection.  
- **collection_type**: (OPTIONAL) The type of collection (default is `'user'`).

#### Example Requests

**Python Example:**

```python
import requests, json
from copy import deepcopy

# Input data with user arguments
input = deepcopy(USER_ARGS_1)  # Substitute USER_ARGS_1 with actual user arguments
input.update({
    "auth": {"api_key": "example_api_key"},
    "collection_hash_id": "your_collection_hash_id",
    "title": "test_collection_1_modified",
    "description": "test description please ignore"
})

response = requests.get("http://localhost:8000/api/modify_document_collection", json=input)
result = response.json()

print(json.dumps(result, indent=4))
```

**JavaScript Example:**

```javascript
const axios = require('axios');

const input = {
    auth: { api_key: "example_api_key" },
    collection_hash_id: "your_collection_hash_id",
    title: "test_collection_1_modified",
    description: "test description please ignore"
};

axios.get("http://localhost:8000/api/modify_document_collection", { data: input })
    .then(response => console.log(JSON.stringify(response.data, null, 4)))
    .catch(error => console.error(error));
```

### Response Structure

The response you receive will contain a JSON object indicating the success of the operation:

```json
{
    "success": true
}
```

In case of an error, an error message may be provided, but the structure of this response is not specified.

**Note:** Ensure that you substitute `your_collection_hash_id` and `USER_ARGS_1` with the actual values needed for your requests.