# API Documentation for Login Endpoint

### ✅ `login`

The `login` endpoint is used for verifying a user's login credentials and providing them with their password prehash. It accepts authentication using one of the supported methods: API key, OAuth2, or username/password.

#### Endpoint
```
GET /api/login
POST /api/login
```

#### Authentication Methods
- **OAuth2**:
    ```json
    {"auth": "oauth2_string"}
    ```

- **Username and Password**:
    ```json
    {"auth": {"username": "example_username", "password": "example_password"}}
    ```

#### Example Usage

**Python Example:**
```python
import requests, json

# Login credentials
login_input = {
    "auth": {
        "username": "example_username",
        "password": "example_password"
    }
}

# Make the login request
response = requests.get("http://localhost:8000/api/login", json=login_input)
result = response.json()

# Print the result
print(json.dumps(result, indent=4))
```

**JavaScript Example:**
```javascript
const axios = require('axios');

// Login credentials
const loginInput = {
    auth: {
        username: "example_username",
        password: "example_password"
    }
};

// Make the login request
axios.get('http://localhost:8000/api/login', { data: loginInput })
    .then(response => {
        console.log(JSON.stringify(response.data, null, 4));
    })
    .catch(error => {
        console.error(error);
    });
```

#### Response Structure
The response from a successful login will resemble the following JSON structure:
```json
{
    "success": true,
    "result": {
        "username": "637e2f0f-454b-4470-824c-708c4577",
        "auth": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
        "memberships": [],
        "admin": false,
        "available_models": {
            "default_models": {
                "llm": "llama-3.1-8b-instruct",
                "rerank": "bge-reranker-v2-m3",
                "embedding": "bge-m3"
            },
            ...
        }
    }
}
```

In case of failure, an error message will be returned indicating the issue with the login attempt.