# API Documentation: `add_user`

### ✅ `add_user`

**Endpoint:** `/api/add_user`  
**Method:** `GET` or `POST`  
**Description:** Add a user to the database. Depending on your user configuration, you can send a confirmation email for signup, and communicate this in the response via `{"pending_email": True}`.

### Authentication

This endpoint supports the following authentication methods:
- **API Key**:
  ```json
  {"auth": {"api_key": "example_api_key"}}
  ```
- **OAuth2**:
  ```json
  {"auth": "oauth2_string"}
  ```
- **Username and Password**:
  ```json
  {"auth": {"username": "example_username", "password": "example_password"}}
  ```

### Request Body Parameters
- `username` (str): The desired username for the new user.
- `password` (str): The password for the new user.

### Example Usage

#### Python Example

```python
import requests
import json

# API Key Authentication
api_key = "example_api_key"
add_user_input = {
    "username": "new_user1",
    "password": "secure_password1",
}

response = requests.post("http://localhost:8000/api/add_user", json={"auth": {"api_key": api_key}, **add_user_input})
result = response.json()
print(json.dumps(result, indent=4))

add_user_input = {
    "username": "new_user2",
    "password": "secure_password2",
}

response = requests.post("http://localhost:8000/api/add_user", json={"auth": {"api_key": api_key}, **add_user_input})
result = response.json()
print(json.dumps(result, indent=4))
```

#### JavaScript Example

```javascript
const axios = require('axios');

// API Key Authentication
const apiKey = "example_api_key";
const addUserInput1 = {
    username: "new_user1",
    password: "secure_password1"
};

axios.post("http://localhost:8000/api/add_user", {
    auth: { api_key: apiKey },
    ...addUserInput1
})
.then(response => {
    console.log(JSON.stringify(response.data, null, 4));
})
.catch(error => {
    console.error(error);
});

const addUserInput2 = {
    username: "new_user2",
    password: "secure_password2"
};

axios.post("http://localhost:8000/api/add_user", {
    auth: { api_key: apiKey },
    ...addUserInput2
})
.then(response => {
    console.log(JSON.stringify(response.data, null, 4));
})
.catch(error => {
    console.error(error);
});
```

This documentation provides guidance on how to use the `add_user` endpoint effectively, showcasing the required parameters, authentication methods, and example requests in both Python and JavaScript.