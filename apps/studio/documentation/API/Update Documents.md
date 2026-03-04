# API Documentation for `update_documents`

### ✅ `update_documents`

Update document metadata and/or text.

#### Description
Set the text for a given document in the database, and perform text chunking, and optionally embedding. You can upload a zipped JSONL file or a data field with the same content. If uploading a zip/7zip file, the file must be named `metadata.jsonl`. Each entry must match the following scheme:

```python
document_id: str
scan: Optional[bool] = False
text: Optional[str] = None
metadata: Optional[Dict[str, Any]] = None
```

- `scan`: Indicates whether to start scanning the file to extract text, chunk it, and put it into the database.
- `text`: Allows you to manually set the text of the document and takes priority over `scan` if it is provided.
- `metadata`: Allows you to set the metadata of the document.

#### Endpoint
```
POST /api/update_documents
GET /api/update_documents
```

### Authentication
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

### Request Body
- `data`: A list of dictionaries containing document updates (optional).
- `file`: A file upload containing metadata (optional).
- `create_embeddings`: A boolean indicating whether to create embeddings (default is `True`).
- `await_embedding`: A boolean indicating whether to await embeddings before responding (default is `False`).

### Example Code

#### Python Example
```python
import requests
import json
import io

# Prepare the data to update the document
update_field = {
    "document_id": "doc123",
    "metadata": {
        "test_field": "test_value"
    },
    "text": "TEST TEST TEST TEST TEST"
}

all_entries = [update_field] # You can add as many as you want here.

# Convert the data to JSONL format and prepare for upload
jsonl_bytes = io.BytesIO()
jsonl_bytes.write("\n".join([json.dumps(e) for e in all_entries]).encode())
jsonl_bytes.seek(0)
jsonl_bytes.name = "metadata.jsonl"

# API parameters
args_file_version = {
    "auth": {"api_key": "example_api_key"},
    "await_embedding": True
}

files = {'file': jsonl_bytes}
encoded_params = f"parameters={json.dumps(args_file_version)}"
response = requests.post(f"http://localhost:8000/update_documents?{encoded_params}", files=files)
result = response.json()

print(json.dumps(result, indent=4))
```

#### JavaScript Example
```javascript
const axios = require('axios');

const updateField = {
    document_id: "doc123",
    metadata: {
        test_field: "test_value"
    },
    text: "TEST TEST TEST TEST TEST"
};

const allEntries = [updateField];

// Convert entries to JSONL format
const jsonlContent = allEntries.map(e => JSON.stringify(e)).join('\n');
const jsonBlob = new Blob([jsonlContent], { type: 'application/jsonl' });
const formData = new FormData();
formData.append("file", jsonBlob, "metadata.jsonl");

// API parameters
const apiParams = {
    auth: { api_key: "example_api_key" },
    await_embedding: true
};

// Make a post request
axios.post(`http://localhost:8000/update_documents?parameters=${encodeURIComponent(JSON.stringify(apiParams))}`, formData)
    .then(response => {
        console.log(JSON.stringify(response.data, null, 4));
    })
    .catch(error => {
        console.error(error);
    });
```

### Expected Response
The response from this endpoint will be in the following format:

```json
{
    "success": true,
    "result": true
}
``` 

This indicates the success of the update operation. If there are any errors, additional information will be included in the response.