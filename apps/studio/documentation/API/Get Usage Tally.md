# API Documentation: `get_usage_tally`

The `get_usage_tally` API endpoint allows users to retrieve usage statistics based on a specified time window and timestamp range. This endpoint supports three authentication methods (API Key, OAuth2, or Username and Password).

## Endpoint
```
POST /api/get_usage_tally
GET /api/get_usage_tally
```

## Authentication
This endpoint requires one of the following authentication methods:

- **API Key**
    ```json
    {"auth": {"api_key": "example_api_key"}}
    ```
- **OAuth2**
    ```json
    {"auth": "oauth2_string"}
    ```
- **Username and Password**
    ```json
    {"auth": {"username": "example_username", "password": "example_password"}}
    ```

## Parameters
| Parameter         | Type     | Description                                      |
|-------------------|----------|--------------------------------------------------|
| `auth`            | QUERYLAKE_AUTH | Authentication details (see above)             |
| `window`          | string   | The time window for retrieving usage data. (e.g., "day") |
| `start_timestamp` | int      | The start timestamp for the usage data query.  |
| `end_timestamp`   | int      | The end timestamp for the usage data query. (optional) |

### Example Request (Python)
```python
import requests
import time
import json

response = requests.get("http://localhost:8000/api/get_usage_tally", json={
    "auth": {"api_key": "example_api_key"},
    "window": "day",
    "start_timestamp": 1722470400,
    "end_timestamp": int(time.time())+1000
})
result = response.json()

print(json.dumps(result, indent=4))
```

### Example Request (JavaScript)
```javascript
const apiKey = "example_api_key";
const startTimestamp = 1722470400;
const endTimestamp = Math.floor(Date.now() / 1000) + 1000;

fetch("http://localhost:8000/api/get_usage_tally", {
    method: "GET",
    body: JSON.stringify({
        auth: { api_key: apiKey },
        window: "day",
        start_timestamp: startTimestamp,
        end_timestamp: endTimestamp
    }),
    headers: {
        "Content-Type": "application/json"
    }
})
.then(response => response.json())
.then(data => console.log(JSON.stringify(data, null, 4)))
.catch(error => console.error('Error:', error));
```

## Response Structure
The response from the API will contain a success flag and the result data.

Example Response:
```json
{
    "success": true,
    "result": [
        {
            "window": "day",
            "api_key_id": "DY5Rq6HFTt8tsUfgSMNNpvfoY0ez6M0W",
            "value": {
                "llm": {
                    "llama-3.1-8b-instruct": {
                        "input_tokens": 6806,
                        "output_tokens": 2677
                    }
                },
                "rerank": {
                    "bge-reranker-v2-m3": {
                        "tokens": 4441
                    }
                },
                "embedding": {
                    "bge-m3": {
                        "tokens": 86
                    }
                }
            },
            "organization_id": null,
            "id": "R35MMYs1S51DE4rty3qR4aoZQ7U81Hux",
            "start_timestamp": 1728345600,
            "user_id": "qJHQLnqANyAHuQviYcf39rPfGpBatRxX"
        }
    ]
}
```

## Notes
- Both `GET` and `POST` requests are supported.
- The `end_timestamp` parameter is optional; if not provided, the server will use the current time as the end of the window.