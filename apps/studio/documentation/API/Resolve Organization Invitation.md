# API Documentation for `resolve_organization_invitation`

### ✅ `resolve_organization_invitation`

**Description**: This endpoint allows a user to accept or decline an invitation to join an organization. It checks the membership status between the user and the organization based on the provided organization ID.

**Endpoint**: `/api/resolve_organization_invitation`

**Supported Methods**: `GET`, `POST`

### Authentication

The following authentication methods are supported:

1. **API Key**:
   ```json
   {"auth": {"api_key": "example_api_key"}}
   ```

2. **OAuth2**:
   ```json
   {"auth": "oauth2_string"}
   ```

3. **Username + Password**:
   ```json
   {"auth": {"username": "example_username", "password": "example_password"}}
   ```

### Request Parameters

- `auth`: Authentication credentials (required, see above)
- `organization_id`: The ID of the organization (required, type: int)
- `accept`: A boolean indicating whether to accept the invitation (required, type: bool)

### Example Request in Python

Here's an example to show how to use the `resolve_organization_invitation` API in Python:

```python
import requests, json

input = {
    "auth": {"api_key": "example_api_key"},
    "organization_id": 123,
    "accept": True
}

response = requests.get("http://localhost:8000/api/resolve_organization_invitation", json=input)
result = response.json()

print(json.dumps(result, indent=4))
```

### Example Request in JavaScript

The following is a similar example for making the same request using JavaScript:

```javascript
const axios = require('axios');

const input = {
    auth: { api_key: "example_api_key" },
    organization_id: 123,
    accept: true
};

axios.get("http://localhost:8000/api/resolve_organization_invitation", { data: input })
    .then(response => {
        console.log(JSON.stringify(response.data, null, 4));
    })
    .catch(error => {
        console.error(error);
    });
```

### Response Structure

A successful response will return a JSON object with the following structure:

```json
{
    "success": true
}
```

Make sure to adapt the `organization_id` and `accept` parameters based on your specific use case. This API call is essential for managing invitations to organizations within your application.