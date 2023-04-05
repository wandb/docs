# Run



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L1664-L2276)



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



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L1761-L1801)

```python
@classmethod
create(
 api, run_id=None, project=None, entity=None
)
```

Create a run for the given project.


### `delete`



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L1914-L1946)

```python
delete(
 delete_artifacts=(False)
)
```

Delete the given run from the wandb backend.


### `display`



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L979-L990)

```python
display(
 height=420, hidden=(False)
) -> bool
```

Display this object in jupyter.


### `file`



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L2009-L2019)

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



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L1996-L2007)

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



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L2045-L2085)

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



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L1803-L1861)

```python
load(
 force=(False)
)
```




### `log_artifact`



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L2183-L2215)

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



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L2142-L2144)

```python
logged_artifacts(
 per_page=100
)
```




### `save`



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L1948-L1949)

```python
save()
```




### `scan_history`



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L2087-L2140)

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



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L975-L977)

```python
snake_to_camel(
 string
)
```




### `to_html`



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L2262-L2270)

```python
to_html(
 height=420, hidden=(False)
)
```

Generate HTML containing an iframe displaying this run.


### `update`



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L1886-L1912)

```python
update()
```

Persist changes to the run object to the wandb backend.


### `upload_file`



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L2021-L2043)

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



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L2150-L2181)

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



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L2146-L2148)

```python
used_artifacts(
 per_page=100
)
```




### `wait_until_finished`



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/apis/public.py#L1863-L1884)

```python
wait_until_finished()
```






