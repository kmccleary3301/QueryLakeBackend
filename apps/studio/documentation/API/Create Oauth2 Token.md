# API Documentation for `create_oauth2_token`

### ✅ `create_oauth2_token`

Create an OAuth2 token for logging in. Note that all OAuth2 tokens will be invalid if the backend is restarted.

#### Endpoint
```
POST /api/create_oauth2_token
```

#### Authentication
The `auth` parameter requires username and password authentication. It must be provided in the following format:
```json
{"auth": {"username": "example_username", "password": "example_password"}}
```

#### Request Example

##### Python
```python
import requests
import json

# Example credentials
input = {"auth": {"username": "example_username", "password": "example_password"}}

response = requests.post("http://localhost:8000/api/create_oauth2_token", json=input)
result = response.json()

print(json.dumps(result, indent=4))
OAUTH2_TOKEN = result["result"]
```

##### JavaScript
```javascript
const fetch = require('node-fetch');

let input = {
    auth: {
        username: "example_username",
        password: "example_password"
    }
};

fetch("http://localhost:8000/api/create_oauth2_token", {
    method: "POST",
    body: JSON.stringify(input),
    headers: { 'Content-Type': 'application/json' }
})
    .then(response => response.json())
    .then(result => {
        console.log(JSON.stringify(result, null, 4));
        const OAUTH2_TOKEN = result.result;
    });
```

#### Response Structure
The response will return a JSON object similar to the following:
```json
{
    "success": true,
    "result": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

#### Notes
- If the request fails, the JSON response may contain an `"error"` field detailing the issue.
- Remember that the OAuth2 token will become invalidated if the backend server is restarted.