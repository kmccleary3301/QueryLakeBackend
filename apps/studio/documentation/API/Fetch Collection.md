# API Documentation for `fetch_collection`

The `fetch_collection` API endpoint retrieves details of a collection for a user, including all documents in the collection. 

## Endpoint
```
GET /api/fetch_collection
```

## Authentication
This endpoint supports three types of authentication (only one may be used at a time):
- **API Key:** 
    ```json
    {"auth": {"api_key": "example_api_key"}}
    ```
- **Username and Password:**
    ```json
    {"auth": {"username": "example_username", "password": "example_password"}}
    ```
- **OAuth2:**
    ```json
    {"auth": "oauth2_string"}
    ```

## Parameters
- `auth` (required): The authentication method as detailed above.
- `collection_hash_id` (required): A string that represents the unique identifier of the collection you wish to fetch.
- `collection_type` (optional): A string that indicates the type of collection. The default value is `'user'`.

## Request Example

### Python Example
Here is a simplified example of how to call `fetch_collection` using Python:

```python
import requests, json

input = {
    "auth": {"api_key": "example_api_key"},  # Use your actual API Key
    "collection_hash_id": "your_collection_hash_id",  # Replace with your collection hash ID
    "collection_type": "user"  # Optional
}

response = requests.get("http://localhost:8000/api/fetch_collection", json=input)
result = response.json()

print(json.dumps(result, indent=4))
```

### JavaScript Example
This is how you would call the same endpoint using JavaScript:

```javascript
const fetchCollection = async () => {
    const input = {
        auth: { api_key: "example_api_key" },  // Use your actual API Key
        collection_hash_id: "your_collection_hash_id",  // Replace with your collection hash ID
        collection_type: "user"  // Optional
    };

    const response = await fetch("http://localhost:8000/api/fetch_collection", {
        method: "GET",
        body: JSON.stringify(input),
        headers: { 'Content-Type': 'application/json' }
    });

    const result = await response.json();
    console.log(JSON.stringify(result, null, 4));
};

fetchCollection();
```

## Response Structure
On a successful request, the API will return a JSON response that includes the following structure:

```json
{
    "success": true,
    "result": {
        "title": "rag_collection",
        "description": "",
        "type": "user",
        "owner": "personal",
        "public": false,
        "document_count": 0
    }
}
```

Check for `success` in the response to verify that the operation completed without issues. If `success` is `false`, an `error` message will provide more context regarding the problem.