# Run



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L1639-L2247)



A single run associated with an entity and project.

```python
Run(
 client: "RetryingClient",
 entity: str,
 project: str,
 run_id: str,
 attrs: Optional[Mapping] = None,
 include_sweeps: bool = (True)
)
```







| Attributes | |
| :--- | :--- |



## Methods

### `create`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L1736-L1776)

```python
@classmethod
create(
 api, run_id=None, project=None, entity=None
)
```

Create a run for the given project.


### `delete`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L1889-L1917)

```python
delete(
 delete_artifacts=(False)
)
```

Delete the given run from the wandb backend.


### `display`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L954-L965)

```python
display(
 height=420, hidden=(False)
) -> bool
```

Display this object in jupyter.


### `file`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L1980-L1990)

```python
file(
 name
)
```

Return the path of a file with a given name in the artifact.


| Arguments | |
| :--- | :--- |
| name (str): name of requested file. |



| Returns | |
| :--- | :--- |
| A `File` matching the name argument. |



### `files`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L1967-L1978)

```python
files(
 names=None, per_page=50
)
```

Return a file path for each file named.


| Arguments | |
| :--- | :--- |
| names (list): names of the requested files, if empty returns all files per_page (int): number of results per page. |



| Returns | |
| :--- | :--- |
| A `Files` object, which is an iterator over `File` objects. |



### `history`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L2016-L2056)

```python
history(
 samples=500, keys=None, x_axis="_step", pandas=(True), stream="default"
)
```

Return sampled history metrics for a run.

This is simpler and faster if you are ok with the history records being sampled.

| Arguments | |
| :--- | :--- |
| `samples` | (int, optional) The number of samples to return |
| `pandas` | (bool, optional) Return a pandas dataframe |
| `keys` | (list, optional) Only return metrics for specific keys |
| `x_axis` | (str, optional) Use this metric as the xAxis defaults to _step |
| `stream` | (str, optional) "default" for metrics, "system" for machine metrics |



| Returns | |
| :--- | :--- |
| `pandas.DataFrame` | If pandas=True returns a `pandas.DataFrame` of history metrics. list of dicts: If pandas=False returns a list of dicts of history metrics. |



### `load`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L1778-L1836)

```python
load(
 force=(False)
)
```




### `log_artifact`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L2154-L2186)

```python
log_artifact(
 artifact, aliases=None
)
```

Declare an artifact as output of a run.


| Arguments | |
| :--- | :--- |
| artifact (`Artifact`): An artifact returned from `wandb.Api().artifact(name)` aliases (list, optional): Aliases to apply to this artifact |



| Returns | |
| :--- | :--- |
| A `Artifact` object. |



### `logged_artifacts`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L2113-L2115)

```python
logged_artifacts(
 per_page=100
)
```




### `save`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L1919-L1920)

```python
save()
```




### `scan_history`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L2058-L2111)

```python
scan_history(
 keys=None, page_size=1000, min_step=None, max_step=None
)
```

Returns an iterable collection of all history records for a run.


#### Example:

Export all the loss values for an example run

```python
run = api.run("l2k2/examples-numpy-boston/i0wt6xua")
history = run.scan_history(keys=["Loss"])
losses = [row["Loss"] for row in history]
```




| Arguments | |
| :--- | :--- |
| keys ([str], optional): only fetch these keys, and only fetch rows that have all of keys defined. page_size (int, optional): size of pages to fetch from the api |



| Returns | |
| :--- | :--- |
| An iterable collection over history records (dict). |



### `snake_to_camel`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L950-L952)

```python
snake_to_camel(
 string
)
```




### `to_html`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L2233-L2241)

```python
to_html(
 height=420, hidden=(False)
)
```

Generate HTML containing an iframe displaying this run.


### `update`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L1861-L1887)

```python
update()
```

Persist changes to the run object to the wandb backend.


### `upload_file`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L1992-L2014)

```python
upload_file(
 path, root="."
)
```

Upload a file.


| Arguments | |
| :--- | :--- |
| path (str): name of file to upload. root (str): the root path to save the file relative to. i.e. If you want to have the file saved in the run as "my_dir/file.txt" and you're currently in "my_dir" you would set root to "../". |



| Returns | |
| :--- | :--- |
| A `File` matching the name argument. |



### `use_artifact`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L2121-L2152)

```python
use_artifact(
 artifact, use_as=None
)
```

Declare an artifact as an input to a run.


| Arguments | |
| :--- | :--- |
| artifact (`Artifact`): An artifact returned from `wandb.Api().artifact(name)` use_as (string, optional): A string identifying how the artifact is used in the script. Used to easily differentiate artifacts used in a run, when using the beta wandb launch feature's artifact swapping functionality. |



| Returns | |
| :--- | :--- |
| A `Artifact` object. |



### `used_artifacts`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L2117-L2119)

```python
used_artifacts(
 per_page=100
)
```




### `wait_until_finished`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L1838-L1859)

```python
wait_until_finished()
```






