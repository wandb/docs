---
title: File
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/files.py#L229-L411 >}}

File saved to W&B.

Represents a single file stored in W&B. Includes access to file metadata.
Files are associated with a specific run and
can include text files, model weights, datasets, visualizations, and other
artifacts. You can download the file, delete the file, and access file
properties.

Specify one or more attributes in a dictionary to fine a specific
file logged to a specific run. You can search using the following keys:

- id (str): The ID of the run that contains the file
- name (str): Name of the file
- url (str): path to file
- direct_url (str): path to file in the bucket
- sizeBytes (int): size of file in bytes
- md5 (str): md5 of file
- mimetype (str): mimetype of file
- updated_at (str): timestamp of last update
- path_uri (str): path to file in the bucket, currently only available for S3 objects and reference files

| Args |  |
| :--- | :--- |
|  `client` |  The run object that contains the file attrs (dict): A dictionary of attributes that define the file |
|  `run` |  The run object that contains the file |

<!-- lazydoc-ignore-init: internal -->


| Attributes |  |
| :--- | :--- |
|  `path_uri` |  Returns the URI path to the file in the storage bucket. |
|  `size` |  Returns the size of the file in bytes. |

## Methods

### `delete`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/files.py#L340-L371)

```python
delete()
```

Delete the file from the W&B server.

### `display`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/attrs.py#L18-L38)

```python
display(
    height=420, hidden=(False)
) -> bool
```

Display this object in jupyter.

### `download`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/files.py#L296-L338)

```python
download(
    root: str = ".",
    replace: bool = (False),
    exist_ok: bool = (False),
    api: (Api | None) = None
) -> io.TextIOWrapper
```

Downloads a file previously saved by a run from the wandb server.

| Args |  |
| :--- | :--- |
|  `root` |  Local directory to save the file. Defaults to the current working directory ("."). |
|  `replace` |  If `True`, download will overwrite a local file if it exists. Defaults to `False`. |
|  `exist_ok` |  If `True`, will not raise ValueError if file already exists and will not re-download unless replace=True. Defaults to `False`. |
|  `api` |  If specified, the `Api` instance used to download the file. |

| Raises |  |
| :--- | :--- |
|  `ValueError` if file already exists, `replace=False` and `exist_ok=False`. |

### `snake_to_camel`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/attrs.py#L14-L16)

```python
snake_to_camel(
    string
)
```

### `to_html`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/attrs.py#L40-L41)

```python
to_html(
    *args, **kwargs
)
```
