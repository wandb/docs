---
title: files
object_type: public_apis_namespace
data_type_classification: module
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/apis/public/files.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API for File objects. 

This module provides classes for interacting with files stored in W&B. 



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

