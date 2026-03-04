# API Documentation: `fetch_memberships`

### ✅ `fetch_memberships`

#### Description
The `fetch_memberships` endpoint returns a list of dictionaries representing organizations for which the user holds a membership. The `return_subset` parameter specifies the type of memberships to return, which can be either "accepted", "open_invitations", or "all".

#### Endpoint
```
GET /api/fetch_memberships
```

#### Function Arguments
- **auth**: (required) Authentication credential. This can be provided as:
  - OAuth2 token: `{"auth": "oauth2_string"}`
  - API Key: `{"auth": {"api_key": "example_api_key"}}`
  - Username and Password: `{"auth": {"username": "example_username", "password": "example_password"}}`
  
- **return_subset**: (optional) Specifies what type of memberships to return. Default value is "accepted". Acceptable values are:
  - `"accepted"`
  - `"open_invitations"`
  - `"all"`

#### Example Request
```json
{
    "auth": {"api_key": "example_api_key"},
    "return_subset": "all"
}
```

#### Example Response
```json
{
    "success": true,
    "result": {
        "memberships": [
            {
                "organization_id": "9UBAXSpoDiprOnbw7YwcZe9kdSX4EL2b",
                "organization_name": "test_org",
                "role": "member",
                "invite_still_open": true,
                "sender": "439dcbc6-316e-454e-9665-0c8ff5c8"
            }
        ],
        "admin": false
    }
}
```

#### Python Code Example
```python
import requests, json
from copy import deepcopy

input = deepcopy({"auth": {"api_key": "example_api_key"}})
input.update({
    "return_subset": "all"
})

response = requests.get("http://localhost:8000/api/fetch_memberships", json=input)
result = response.json()

print(json.dumps(result, indent=4))
```

#### JavaScript Code Example
```javascript
const axios = require('axios');

const input = {
    auth: { api_key: 'example_api_key' },
    return_subset: 'all'
};

axios.get('http://localhost:8000/api/fetch_memberships', { data: input })
    .then(response => {
        console.log(JSON.stringify(response.data, null, 4));
    })
    .catch(error => {
        console.error(error);
    });
```

### Notes
- Ensure that the correct authentication method is used based on your application's requirements.
- The success flag in the response indicates whether the request was successful or not.