# API Documentation: `upload_document`

## Endpoint
**`POST /api/upload_document`**

## Description
Upload a file to the server. This file can be a user document, organization document, or a toolchain session document. If uploading for a toolchain session, provide the toolchain session hash ID as the `collection_hash_id`.

## Authentication
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

## Function Arguments

| Keyword               | Type hint                                              | Default        | Description                                                     |
|----------------------|-------------------------------------------------------|----------------|-----------------------------------------------------------------|
| `auth`               | `Auth (see above)`                                     |                | Authentication object (API key, OAuth2, or username/password). |
| `file`               | `BytesIO` |                | The file to be uploaded.                                       |
| `collection_hash_id` | `str`                                                |                | Hash ID of the collection where the document should be stored.  |
| `file_name`          | `str`                                                | `None`         | Optional name for the file being uploaded.                    |
| `collection_type`    | `str`                                                | `'user'`       | Type of collection to associate the document with.            |
| `return_file_hash`   | `bool`                                              | `False`        | Should the response include the file hash?                    |
| `create_embeddings`   | `bool`                                              | `True`         | Should embeddings be created for the document?                   |
| `await_embedding`     | `bool`                                              | `False`        | Should the upload wait for embeddings to be processed?          |
| `document_metadata`     | `dict`                                              | `None`        | Add custom metadata to the document that will be copied to all the text chunks |

## Example Usage

### Python Example
```python
import requests, json
from urllib.parse import urlencode

# Define your authentication method
auth = {"api_key": "example_api_key"}

# Get the id of your collection to upload to.
collection_hash_id = "your_collection_hash_id"

# Pick your file path
pdf_file_path = "path/to/first.pdf"

# Prepare the input parameters
input_data = {
    "auth": auth,
    "collection_hash_id": collection_hash_id,
    "await_embedding": True
}


with open(pdf_file_path, 'rb') as f:
    files = {'file': f}
    new_inputs = {**input_data, "create_embeddings": (i != 2)}
    encoded_params = urlencode({"parameters": json.dumps(new_inputs)})
    
    response = requests.post(f"http://localhost:8000/upload_document?{encoded_params}", files=files)
    result = response.json()

    print(json.dumps(result, indent=4))
```

### JavaScript Example
```javascript
const axios = require('axios');
const fs = require('fs');
const querystring = require('querystring');

const auth = { "auth": { "api_key": "example_api_key" } };
const collectionHashId = "your_collection_hash_id";
const pdfFiles = ["path/to/first.pdf", "path/to/second.pdf", "path/to/large.pdf"];

const inputData = {
    "collection_hash_id": collectionHashId,
    "await_embedding": true
};

pdfFiles.forEach((path, index) => {
    const formData = new FormData();
    const file = fs.createReadStream(path);
    formData.append('file', file);

    const newInputs = {
        ...inputData,
        create_embeddings: (index !== 2)
    };

    const encodedParams = querystring.stringify({ parameters: JSON.stringify(newInputs) });

    axios.post(`http://localhost:8000/upload_document?${encodedParams}`, formData, {
        headers: formData.getHeaders()
    })
    .then(response => {
        console.log(JSON.stringify(response.data, null, 4));
    })
    .catch(error => {
        console.error(error);
    });
});
```

## Response Format
Upon a successful upload, the server will return a JSON response with the following structure:
```json
{
    "success": true,
    "result": {
        "hash_id": "vBdEyuBoqhw3FV2eLa1HXPcCjfGcKj0Q",
        "title": "stats_book.pdf",
        "size": "7.1 MB",
        "finished_processing": true
    }
}
```

This includes information about the uploaded file like its hash ID, title, size, and whether the processing is finished.