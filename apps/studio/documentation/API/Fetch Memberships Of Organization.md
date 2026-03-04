# API Documentation for `fetch_memberships_of_organization`

## ✅ `fetch_memberships_of_organization`

### Description
Fetches all active memberships of an organization, first verifying that the user is in the given organization.

### Endpoint
```
GET /api/fetch_memberships_of_organization
```

### Required Arguments
- `auth`: Authentication information (see [Authentication](#authentication))
- `organization_id`: The ID of the organization (integer)

### Authentication
There are three methods of authentication for this endpoint. You can use any one of the following:

1. **API Key**
   ```json
   {"auth": {"api_key": "example_api_key"}}
   ```
   
2. **Username and Password**
   ```json
   {"auth": {"username": "example_username", "password": "example_password"}}
   ```

3. **OAuth2**
   ```json
   {"auth": "oauth2_string"}
   ```

### Example Requests

#### Python Example
```python
import requests, json

input = {
    "auth": {"api_key": "example_api_key"},
    "organization_id": 123456
}

response = requests.get("http://localhost:8000/api/fetch_memberships_of_organization", json=input)
result = response.json()

print(json.dumps(result, indent=4))
```

#### JavaScript Example
```javascript
const axios = require('axios');

const input = {
    auth: { api_key: "example_api_key" },
    organization_id: 123456
};

axios.get("http://localhost:8000/api/fetch_memberships_of_organization", { data: input })
    .then(response => {
        console.log(JSON.stringify(response.data, null, 4));
    })
    .catch(error => {
        console.error(error);
    });
```

### Expected Output
The API will return a JSON object on success, in the following structure:

```json
{
    "success": true,
    "result": {
        "memberships": [
            {
                "organization_id": "9UBAXSpoDiprOnbw7YwcZe9kdSX4EL2b",
                "organization_name": "test_org",
                "role": "owner",
                "username": "439dcbc6-316e-454e-9665-0c8ff5c8",
                "invite_still_open": false
            },
            {
                "organization_id": "9UBAXSpoDiprOnbw7YwcZe9kdSX4EL2b",
                "organization_name": "test_org",
                "role": "member",
                "username": "637e2f0f-454b-4470-824c-708c4577",
                "invite_still_open": false
            }
        ]
    }
}
```

In case of an error, the response may look like this:

```json
{
    "success": false,
    "error": "Error message here"
}
``` 

Feel free to use this endpoint to manage and view memberships within your organizations!