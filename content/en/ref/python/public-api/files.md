---
title: Files
namespace: public_apis_namespace
python_object_type: class
---
{{< readfile file="/_includes/public-api-use.md" >}}


{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/files.py >}}




## <kbd>class</kbd> `Files`
A lazy iterator over a collection of `File` objects. 

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
__init__(
    client: 'RetryingClient',
    run: 'Run',
    names: 'list[str] | None' = None,
    per_page: 'int' = 50,
    upload: 'bool' = False,
    pattern: 'str | None' = None
)
```

Initialize a lazy iterator over a collection of `File` objects. 

Files are retrieved in pages from the W&B server as needed. 



**Args:**
 client: The run object that contains the files run: The run object that contains the files names (list, optional): A list of file names to filter the files per_page (int, optional): The number of files to fetch per page upload (bool, optional): If `True`, fetch the upload URL for each file pattern (str, optional): Pattern to match when returning files from W&B  This pattern uses mySQL's LIKE syntax,  so matching all files that end with .json would be "%.json".  If both names and pattern are provided, a ValueError will be raised. 


---


### <kbd>property</kbd> Files.length





---



