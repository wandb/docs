---
title: files
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/apis/public/files.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API for File Management. 

This module provides classes for interacting with files stored in W&B. Classes include: 

Files: A paginated collection of files associated with a run 
- Iterate through files with automatic pagination 
- Filter files by name 
- Access file metadata and properties 
- Download multiple files 

File: A single file stored in W&B 
- Access file metadata (size, mimetype, URLs) 
- Download files to local storage 
- Delete files from W&B 
- Work with S3 URIs for direct access 



**Example:**
 ```python
from wandb.apis.public import Api

# Initialize API
api = Api()

# Get files from a specific run
run = api.run("entity/project/run_id")
files = run.files()

# Work with files
for file in files:
     print(f"File: {file.name}")
     print(f"Size: {file.size} bytes")
     print(f"Type: {file.mimetype}")

     # Download file
     if file.size < 1000000:  # Less than 1MB
         file.download(root="./downloads")

     # Get S3 URI for large files
     if file.size >= 1000000:
         print(f"S3 URI: {file.path_uri}")
``` 



**Note:**

> This module is part of the W&B Public API and provides methods to access, download, and manage files stored in W&B. Files are typically associated with specific runs and can include model weights, datasets, visualizations, and other artifacts. 



## <kbd>class</kbd> `Files`
An iterable collection of `File` objects. 

Access and manage files uploaded to W&B during a run. Handles pagination automatically when iterating through large collections of files. 



**Args:**
 
 - `client`:  The run object that contains the files 
 - `run`:  The run object that contains the files 
 - `names` (list, optional):  A list of file names to filter the files 
 - `per_page` (int, optional):  The number of files to fetch per page 
 - `upload` (bool, optional):  If `True`, fetch the upload URL for each file 



**Example:**
 ```python
from wandb.apis.public.files import Files
from wandb.apis.public.api import Api

# Initialize the API client
api = Api()

# Example run object
run = api.run("entity/project/run-id")

# Create a Files object to iterate over files in the run
files = Files(api.client, run)

# Iterate over files
for file in files:
    print(file.name)
    print(file.url)
    print(file.size)

    # Download the file
    file.download(root="download_directory", replace=True)
``` 

### <kbd>method</kbd> `Files.__init__`

```python
__init__(client, run, names=None, per_page=50, upload=False)
```






---

### <kbd>property</kbd> Files.cursor

Returns the cursor position for pagination of file results. 

---

### <kbd>property</kbd> Files.length

The number of files saved to the specified run. 

---

### <kbd>property</kbd> Files.more

Returns `True` if there are more files to fetch. Returns `False` if there are no more files to fetch. 



---

### <kbd>method</kbd> `Files.convert_objects`

```python
convert_objects()
```

Converts GraphQL edges to File objects. 

---

### <kbd>method</kbd> `Files.update_variables`

```python
update_variables()
```

Updates the GraphQL query variables for pagination. 


---

## <kbd>class</kbd> `File`
File saved to W&B. 

Represents a single file stored in W&B. Includes access to file metadata. Files are associated with a specific run and can include text files, model weights, datasets, visualizations, and other artifacts. You can download the file, delete the file, and access file properties. 

Specify one or more attributes in a dictionary to fine a specific file logged to a specific run. You can search using the following keys: 


- id (str): The ID of the run that contains the file 
- name (str): Name of the file 
- url (str): path to file 
- direct_url (str): path to file in the bucket 
- sizeBytes (int): size of file in bytes 
- md5 (str): md5 of file 
- mimetype (str): mimetype of file 
- updated_at (str): timestamp of last update 
- path_uri (str): path to file in the bucket, currently only available for files stored in S3 



**Args:**
 
 - `client`:  The run object that contains the file 
 - `attrs` (dict):  A dictionary of attributes that define the file 
 - `run`:  The run object that contains the file 



**Example:**
 ```python
from wandb.apis.public.files import File
from wandb.apis.public.api import Api

# Initialize the API client
api = Api()

# Example attributes dictionary
file_attrs = {
    "id": "file-id",
    "name": "example_file.txt",
    "url": "https://example.com/file",
    "direct_url": "https://example.com/direct_file",
    "sizeBytes": 1024,
    "mimetype": "text/plain",
    "updated_at": "2025-03-25T21:43:51Z",
    "md5": "d41d8cd98f00b204e9800998ecf8427e",
}

# Example run object
run = api.run("entity/project/run-id")

# Create a File object
file = File(api.client, file_attrs, run)

# Access some of the attributes
print("File ID:", file.id)
print("File Name:", file.name)
print("File URL:", file.url)
print("File MIME Type:", file.mimetype)
print("File Updated At:", file.updated_at)

# Access File properties
print("File Size:", file.size)
print("File Path URI:", file.path_uri)

# Download the file
file.download(root="download_directory", replace=True)

# Delete the file
file.delete()
``` 

### <kbd>method</kbd> `File.__init__`

```python
__init__(client, attrs, run=None)
```






---

### <kbd>property</kbd> File.path_uri

Returns the URI path to the file in the storage bucket. 

---

### <kbd>property</kbd> File.size

Returns the size of the file in bytes. 



---

### <kbd>method</kbd> `File.delete`

```python
delete()
```

Deletes the file from the W&B server. 

---

### <kbd>method</kbd> `File.download`

```python
download(
    root: str = '.',
    replace: bool = False,
    exist_ok: bool = False,
    api: Optional[wandb.apis.public.api.Api] = None
) → TextIOWrapper
```

Downloads a file previously saved by a run from the wandb server. 



**Args:**
 
 - `root`:  Local directory to save the file.  Defaults to ".". 
 - `replace`:  If `True`, download will overwrite a local file  if it exists. Defaults to `False`. 
 - `exist_ok`:  If `True`, will not raise ValueError if file already  exists and will not re-download unless replace=True.  Defaults to `False`. 
 - `api`:  If specified, the `Api` instance used to download the file. 



**Raises:**
 `ValueError` if file already exists, `replace=False` and `exist_ok=False`. 


