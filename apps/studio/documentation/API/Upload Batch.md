# API Documentation: `upload_document` (Batch Upload)

## Endpoint
**`POST /api/upload_document`**

## Description
Upload multiple files to the same collection on the server. It uses the same endpoint as the standard upload, the only difference is that you upload a `.zip` or `.7z` (7zip) archive of your files. This is ideal for reducing overhead when you're uploading a large amount of files. 

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
| `file`               | `BytesIO` |                | The zip file to be uploaded. |
| `collection_hash_id` | `str`                                                |                | Hash ID of the collection where the document should be stored.  |
| `file_name`          | `str`                                                | `None`         | Optional name for the file being uploaded. |
| `collection_type`    | `str`                                                | `'user'`       | Type of collection to associate the document with. |
| `create_embeddings`   | `bool` | `True`         | Should embeddings be created for the document chunks? |
| `await_embedding`     | `bool` | `False`        | Should the upload wait for embeddings to be processed before returning? |
| `scan_text`     | `bool` | `True`        | Scan the document and chunk the text. |

# Example

Let's do an example with batching. First, let's set our information.

```python
QL_API_KEY = "sk_1234"			# Your QueryLake API key
QL_COLLECTION_ID = "abcdefg"	# Your QueryLake collection ID
```

### Define our upload function

```python
import requests
import json
from urllib.parse import urlencode

def send_post_request(file_obj, args : dict):
	files = {'file': file_obj}
	encoded_params = urlencode({"parameters": json.dumps(args)})
	response = requests.post("http://localhost:8000/upload_document?" + encoded_params, files=files)
	response.raise_for_status()

	result = response.json()

	if ("success" in result and result["success"] == False):
		print(result["trace"])

	assert not ("success" in result and result["success"] == False), result["note"]
	return result["result"]
```

### Define a zipping function

This will allow us to construct our archives to send to the post request. The function requires a dictionary, with the keys being file names and the values being file bytes as `io.BytesIO` objects.

```python
import io
from typing import Dict
import zipfile

def zip_file(file_data: Dict[str, io.BytesIO]) -> bytes:
	"""
	Zips a dictionary of file data.
	Returns the zipped file as bytes.
	"""
	new_bytes = io.BytesIO()
	
	with zipfile.ZipFile(new_bytes, 'w', zipfile.ZIP_DEFLATED) as z:
		for filename, file_content in file_data.items():
			z.writestr(filename, file_content.getvalue())
	
	return new_bytes.getvalue()
```

### Load Your Files

The documents in your collection don't necessarily have to have unique names, but when they're encoded in a zip file they do since it's the same as a folder or dictionary. This restriction only applies on a singular batch upload, and you can upload files with duplicate names on another batch upload.

```python
with open("/shared_folders/querylake_server/other/stats_book.pdf", "rb") as f:
    pdf_bytes = f.read()
    f.close()

files = [
	# For text, first encode it to bytes, then initialize a BytesIO object with the bytes
	("file_1_text.txt", io.BytesIO("This is the content of the first file. ".encode())), 	
	("file_2_text.txt", io.BytesIO("This is the content of the second file.".encode())), 	

	# For bytes, initialize a BytesIO object with the bytes
	("stats_book.pdf", io.BytesIO(pdf_bytes))
]
```

### Upload Files

You can't upload more than 10,000 files in one batch, so we'll split it up.

```python
from tqdm import tqdm # Gives us a nice progress bar.

BATCH_SIZE = 2000					# Number of files to send in each batch
current_file_batch_dictionary = {} 	# We will dump this at the end of each batch.


upload_call_inputs = {
    "auth": {"api_key": QL_API_KEY}, 
    "collection_hash_id": QL_COLLECTION_ID,
    "create_embeddings": False, # True by default, if enabled will create embeddings for the document chunks
    
    "await_embedding": True 	# If enabled, will wait until the chunking and embeddings are created before returning the call.
    # Your uploads will be faster if you set this to False, but the server processing will be the same.
}

for file_index, (file_name, file_obj) in tqdm(enumerate(files)):
	current_file_batch_dictionary.update({file_name: file_obj})
	
	if ((file_index + 1) % BATCH_SIZE == 0) or \
		(file_index == len(files) - 1):

		# Convert the current batch to a zip file (bytes)
		zip_file_bytes = zip_file(current_file_batch_dictionary)

		# Create a BytesIO object from the bytes
		zip_file_bytes_io = io.BytesIO(zip_file_bytes)
  
		# Set the file name
		zip_file_bytes_io.name = "blob.zip"
		
		# Send the batch
		send_post_request(zip_file_bytes_io, {**upload_call_inputs})
  
		# Reset the current_file_batch_dictionary
		del current_file_batch_dictionary
		current_file_batch_dictionary = {}
```

Each batch upload call will return the same as a normal file upload, but as a list of these results for each file you uploaded.
Here's an example result:

```json
{
	"success": true,
	"result":
	[
		{
			"hash_id": "UAmXeGHS451fYw3fNh0j6j7e9WFJnTZt",
			"title": "file_1_text.txt",
			"size": "39.0 B",
			"finished_processing": true
		},
		{
			"hash_id": "PdiFDfmjYr9uHFCdlRXnurL9kZiLVfBy",
			"title": "file_2_text.txt",
			"size": "39.0 B",
			"finished_processing": true
		},
		{
			"hash_id": "MutbUU0FvzfWVudcessEz3U6TTSr9OEb",
			"title": "stats_book.pdf",
			"size": "7.1 MB",
			"finished_processing": true
		}
	]
}
```

### Check on our collection

```python
check_args = {
    "auth": {"api_key": QL_API_KEY}, 
	"collection_hash_id": QL_COLLECTION_ID,
}

response = requests.get("http://localhost:8000/api/fetch_collection", json=check_args)
response.raise_for_status()

result = response.json()

print(json.dumps(result, indent=4))
```

```json
{
    "success": true,
    "result": {
        "title": "Batch upload test collection",
        "description": "",
        "type": "user",
        "owner": "personal",
        "public": false,
        "document_count": 3
    }
}
```