---
title: files
object_type: client_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/apis/public/files.py >}}




# <kbd>module</kbd> `wandb.apis.public`
Public API: files. 



## <kbd>class</kbd> `Files`
An iterable collection of `File` objects. 

### <kbd>method</kbd> `Files.__init__`

```python
__init__(client, run, names=None, per_page=50, upload=False)
```






---

### <kbd>property</kbd> Files.cursor





---

### <kbd>property</kbd> Files.length





---

### <kbd>property</kbd> Files.more







---

### <kbd>method</kbd> `Files.convert_objects`

```python
convert_objects()
```





---

### <kbd>method</kbd> `Files.update_variables`

```python
update_variables()
```






---

## <kbd>class</kbd> `File`
File is a class associated with a file saved by wandb. 



**Attributes:**
 
 - `name` (string):  filename 
 - `url` (string):  path to file 
 - `direct_url` (string):  path to file in the bucket 
 - `md5` (string):  md5 of file 
 - `mimetype` (string):  mimetype of file 
 - `updated_at` (string):  timestamp of last update 
 - `size` (int):  size of file in bytes 
 - `path_uri` (str):  path to file in the bucket, currently only available for files stored in S3 

### <kbd>method</kbd> `File.__init__`

```python
__init__(client, attrs, run=None)
```






---

### <kbd>property</kbd> File.path_uri

Returns the uri path to the file in the storage bucket. 

---

### <kbd>property</kbd> File.size







---

### <kbd>method</kbd> `File.delete`

```python
delete()
```





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
 
 - `replace` (boolean):  If `True`, download will overwrite a local file  if it exists. Defaults to `False`. 
 - `root` (str):  Local directory to save the file.  Defaults to ".". 
 - `exist_ok` (boolean):  If `True`, will not raise ValueError if file already  exists and will not re-download unless replace=True. Defaults to `False`. 
 - `api` (Api, optional):  If given, the `Api` instance used to download the file. 



**Raises:**
 `ValueError` if file already exists, replace=False and exist_ok=False. 


