# API Documentation: Invite User to Organization

### ✅ `invite_user_to_organization`

This endpoint is used to invite a user to an organization. If the user is already a member, an error will be raised.

#### Endpoint
```
GET /api/invite_user_to_organization
POST /api/invite_user_to_organization
```

#### Authentication
This endpoint requires one of the following authentication methods:
- **API Key**:
  ```json
  {"auth": {"api_key": "example_api_key"}}
  ```
  
- **Username and Password**:
  ```json
  {"auth": {"username": "example_username", "password": "example_password"}}
  ```

#### Parameters
| Parameter             | Type   | Required | Default Value | Description                                     |
|-----------------------|--------|----------|---------------|-------------------------------------------------|
| `auth`                | Auth (see above) | Yes      | N/A           | The authentication method  |
| `username_to_invite`  | str    | Yes      | N/A           | The username of the user to invite               |
| `organization_id`     | int    | Yes      | N/A           | The ID of the organization                       |
| `member_class`        | str    | No       | 'member'      | The class of membership for the invited user    |

#### Example Usage

##### Python
```python
import requests, json
from copy import deepcopy

# Define your input parameters
input = {
    "auth": {"api_key": "example_api_key"},
    "username_to_invite": "new_user",
    "organization_id": 2,
    "member_class": "member"
}

response = requests.get("http://localhost:8000/api/invite_user_to_organization", json=input)
result = response.json()
print(json.dumps(result, indent=4))
```

##### JavaScript
```javascript
const axios = require('axios');

const input = {
    "auth": {"api_key": "example_api_key"},
    "username_to_invite": "new_user",
    "organization_id": 2,
    "member_class": "member"
};

axios.get("http://localhost:8000/api/invite_user_to_organization", { data: input })
    .then(response => {
        console.log(response.data);
    })
    .catch(error => {
        console.error(error);
    });
```

### Response
A successful response will return a JSON object:
```json
{
    "success": true
}
```

In case of an error, the response may contain:
```json
{
    "success": false,
    "error": "Error message here"
}
```

### Notes
- Ensure that the user you are inviting is not already a member of the organization to avoid errors.
- Adjust the member_class parameter as necessary based on your organization's roles.