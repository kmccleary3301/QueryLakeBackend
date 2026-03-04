# API Documentation for `search_bm25`

### ✅ `search_bm25`

Search over specified document collections using BM25 given a query.

#### Endpoint
```
GET /api/search_bm25
POST /api/search_bm25
```

#### Authentication
This endpoint supports three types of authentication. Only one is required for any request:
1. **API Key**
   ```json
   {"auth": {"api_key": "example_api_key"}}
   ```
2. **Username and Password**
   ```json
   {"auth": {"username": "example_username", "password": "example_password"}}
   ```
3. **OAuth2 Token**
   ```json
   {"auth": "oauth2_string"}
   ```

#### Request Parameters
- `auth` (required): Authentication details (see above).
- `query` (required): The search query string.
- `collection_ids` (optional): List of collection IDs to search over. Defaults to an empty list.
- `limit` (optional): The maximum number of results to return. Defaults to 10.
- `offset` (optional): The number of results to skip for pagination. Defaults to 0.
- `web_search` (optional): Whether to include web search in results. Defaults to False.

#### Example Request

##### Python
```python
import requests

input = {
    "auth": {"api_key": "example_api_key"},
    "query": "artificial neural network",
    "collection_ids": ["FmBtLFCl0ZTwM98imoDtAvRfqZIKycXe"],
    "limit": 10,
}

response = requests.get("http://localhost:8000/api/search_bm25", json=input)
result = response.json()

print(result)
```

##### JavaScript
```javascript
const axios = require('axios');

const input = {
    auth: { api_key: "example_api_key" },
    query: "artificial neural network",
    collection_ids: ["FmBtLFCl0ZTwM98imoDtAvRfqZIKycXe"],
    limit: 10,
};

axios.get("http://localhost:8000/api/search_bm25", { data: input })
    .then(response => {
        console.log(response.data);
    });
```

#### Example Response
```json
{
    "success": true,
    "result": [
        {
            "id": [
                "p2wsIyewrfDsa8f0WabEkFSbvYv03aAg",
                "l6TffURzGtfFJNRMwI4Nb2vsKgNKrgHm"
            ],
            "creation_timestamp": 1728425385.936077,
            "collection_type": "user",
            "document_id": "8JYLF6k9wpnkUbgZdOVowO2XhFXP1v5T",
            "document_chunk_number": [0, 1],
            "collection_id": "FmBtLFCl0ZTwM98imoDtAvRfqZIKycXe",
            "document_name": "HNRS3035_08_22_2023_MLIntro.pdf",
            "md": {
                "page": 1,
                "location_link_chrome": "#page=1&zoom=115,0.0,0.0",
                "location_link_firefox": "#page=1&zoom=115,0.0,612.0"
            },
            "document_md": {
                "file_name": "HNRS3035_08_22_2023_MLIntro.pdf",
                "size_bytes": 493215,
                "integrity_sha256": "f9fedd6b309833c6a40d1a63cc03bc930586308af06504408e103540278c98a0"
            },
            "text": "Introduction to AI/ML: ..."
        }
    ]
}
```

### Notes
- The `add_user` function does not require authentication but may be disabled depending on server configuration.
- The response structure is simplified for example purposes; the actual response may vary.
- Ensure proper integration based on the respective programming environment for better handling of requests and responses.