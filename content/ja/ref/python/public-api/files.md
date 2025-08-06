---
data_type_classification: module
menu:
  reference:
    identifier: ja-ref-python-public-api-files
object_type: public_apis_namespace
title: files
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/files.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API for File objects. 

This module provides classes for interacting with files stored in W&B. 



**Example:**
 ```python
from wandb.apis.public import Api

# Get files from a specific run
run = Api().run("entity/project/run_id")
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



**Example:**
 ```python
from wandb.apis.public.files import Files
from wandb.apis.public.api import Api

# Example run object
run = Api().run("entity/project/run-id")

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

An iterable collection of `File` objects for a specific run. 



**Args:**
 client: The run object that contains the files run: The run object that contains the files names (list, optional): A list of file names to filter the files per_page (int, optional): The number of files to fetch per page upload (bool, optional): If `True`, fetch the upload URL for each file 


---


### <kbd>property</kbd> Files.length





---