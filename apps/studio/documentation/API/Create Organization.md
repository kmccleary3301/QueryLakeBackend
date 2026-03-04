# create_organization API Documentation

### ✅ `create_organization`

The `create_organization` endpoint allows users to add an organization to the database after verifying their identity. Users can authenticate using one of three methods: OAuth2, API Key, or username/password. 

### Endpoint
```
POST /api/create_organization
GET /api/create_organization
```

### Request Structure
The request should contain a JSON body with the following parameters:

- `auth`: (required) Authentication information. Use one of the following formats:
  - For API Key:
    ```json
    {"auth": {"api_key": "example_api_key"}}
    ```
  - For OAuth2:
    ```json
    {"auth": "oauth2_string"}
    ```
  - For username/password:
    ```json
    {"auth": {"username": "example_username", "password": "example_password"}}
    ```

- `organization_name`: (required) Name of the organization to be created.
- `organization_description`: (optional) Description of the organization. Defaults to `None`.

### Example Request
Here’s an example of how to use this endpoint in Python and JavaScript.

#### Python Example
```python
import requests
import json

input_data = {
    "auth": {"api_key": "example_api_key"},
    "organization_name": "test_org",
    "organization_description": "A test organization"
}

response = requests.post("http://localhost:8000/api/create_organization", json=input_data)
result = response.json()

print(json.dumps(result, indent=4))
```

#### JavaScript Example
```javascript
const fetch = require('node-fetch');

const inputData = {
    auth: { api_key: "example_api_key" },
    organization_name: "test_org",
    organization_description: "A test organization"
};

fetch("http://localhost:8000/api/create_organization", {
    method: "POST",
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify(inputData)
})
.then(response => response.json())
.then(result => console.log(JSON.stringify(result, null, 4)))
.catch(error => console.error('Error:', error));
```

### Response Structure
Upon a successful organization creation, the response will contain the following structure:
```json
{
    "success": true,
    "result": {
        "organization_id": "9UBAXSpoDiprOnbw7YwcZe9kdSX4EL2b",
        "organization_dict": {
            "creation_timestamp": 1728425398.4558585,
            "id": "9UBAXSpoDiprOnbw7YwcZe9kdSX4EL2b",
            "serp_api_key_encrypted": null,
            "openai_organization_id_encrypted": null,
            "name": "test_org",
            "public_key": "d6c0fe4825258a09e586920e569fa5c8a3154609ad3aea2b351764ceeb4d8851972be52746afe504525980f14637e475ee2544c31160cef29d6a858b30b9ec84",
            "hash_id": "ea2436009c1e26cc98727ace9f91f76e0d33f8276a5043952a32030469c9decb"
        },
        "membership_dict": {}
    }
}
```
If the organization creation fails, the response will indicate success as `false`, along with error information.

### Note
Make sure to use the appropriate authentication method as specified. The API key is required for most operations, while creating an API key or logging in requires username/password or OAuth2 authentication.