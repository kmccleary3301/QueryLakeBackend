# API Documentation for `search_hybrid`

### ✅ `search_hybrid`

Search over document collections with hybrid bm25+vector search, and optionally with a reranker. You can control the separate queries for bm25, embedding, and reranking if desired.

**Endpoint:** `/api/search_hybrid`

**Methods Supported:** `GET`, `POST`

### Authentication

The `search_hybrid` endpoint requires one of the following authentication methods:

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

### Parameters

| Parameter           | Type                             | Default       | Description                                       |
|---------------------|----------------------------------|---------------|---------------------------------------------------|
| `auth`              | `QUERYLAKE_AUTH`                | -             | Authentication information                         |
| `query`             | `Union[str, dict[str, str]]`    | -             | Search query, can include bm25 and embedding      |
| `embedding`         | `List[float]`                   | `None`       | Optional embedding values                          |
| `collection_ids`    | `List[str]`                     | `[]`          | List of collection IDs to search                  |
| `limit_bm25`        | `int`                           | `10`          | Limit for bm25 results                             |
| `limit_similarity`   | `int`                           | `10`          | Limit for similarity results                       |
| `similarity_weight` | `float`                        | `0.1`         | Weight for similarity in scoring                   |
| `bm25_weight`       | `float`                        | `0.9`         | Weight for bm25 in scoring                         |
| `return_statement`   | `bool`                         | `False`       | Include return statement in the response           |
| `web_search`        | `bool`                         | `False`       | Flag for conducting web search                     |
| `rerank`            | `bool`                         | `False`       | Flag for reranking the results                     |

### Request Example

**Python Example:**
```python
import requests

input = {
    "auth": {"api_key": "your_example_api_key"},
    "query": {
        "bm25": "representation learning",
        "embedding": "representation learning",
        "rerank": "What is representation learning?"
    },
    "collection_ids": ["FmBtLFCl0ZTwM98imoDtAvRfqZIKycXe"],
    "bm25_limit": 10,
    "similarity_limit": 10,
    "bm25_weight": 0.9,
    "similarity_weight": 0.1
}

response = requests.get("http://localhost:8000/api/search_hybrid", json=input)
result = response.json()

print(result)
```

**JavaScript Example:**
```javascript
const input = {
    auth: { api_key: "your_example_api_key" },
    query: {
        bm25: "representation learning",
        embedding: "representation learning",
        rerank: "What is representation learning?"
    },
    collection_ids: ["FmBtLFCl0ZTwM98imoDtAvRfqZIKycXe"],
    bm25_limit: 10,
    similarity_limit: 10,
    bm25_weight: 0.9,
    similarity_weight: 0.1
};

fetch("http://localhost:8000/api/search_hybrid", {
    method: "GET",
    body: JSON.stringify(input),
    headers: {
        "Content-Type": "application/json"
    }
})
.then(response => response.json())
.then(result => {
    console.log(result);
});
```

### Response Structure

The response typically includes:

```json
{
    "success": true,
    "result": [
        {
            "id": ["document_id_1", "document_id_2"],
            "collection_id": "collection_id_example",
            ...
        }
    ]
}
```

### Notes

- Remember to replace `your_example_api_key` with your actual API key.
- Other authentication methods like OAuth2 or Username+Password can be used for login or when creating/deleting an API key.