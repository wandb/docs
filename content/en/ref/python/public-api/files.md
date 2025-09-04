---
title: Files
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.21.3/wandb/apis/public/files.py#L78-L217 >}}

A lazy iterator over a collection of `File` objects.

Access and manage files uploaded to W&B during a run. Handles pagination
automatically when iterating through large collections of files.

#### Example:

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

| Attributes |  |
| :--- | :--- |
|  `cursor` |  Returns the cursor position for pagination of file results. <!-- lazydoc-ignore: internal --> |
|  `more` |  Returns whether there are more files to fetch. <!-- lazydoc-ignore: internal --> |

## Methods

### `convert_objects`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.3/wandb/apis/public/files.py#L206-L214)

```python
convert_objects()
```

Converts GraphQL edges to File objects.

<!-- lazydoc-ignore: internal -->


### `next`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.3/wandb/apis/paginator.py#L102-L109)

```python
next() -> T
```

Return the next item from the iterator. When exhausted, raise StopIteration

### `update_variables`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.3/wandb/apis/public/files.py#L199-L204)

```python
update_variables()
```

Updates the GraphQL query variables for pagination.

<!-- lazydoc-ignore: internal -->


### `__getitem__`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.3/wandb/apis/paginator.py#L95-L100)

```python
__getitem__(
    index: (int | slice)
) -> (T | list[T])
```

### `__iter__`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.3/wandb/apis/paginator.py#L50-L52)

```python
__iter__() -> Iterator[T]
```

### `__len__`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.3/wandb/apis/paginator.py#L128-L133)

```python
__len__() -> int
```

| Class Variables |  |
| :--- | :--- |
|  `QUERY`<a id="QUERY"></a> |   |
