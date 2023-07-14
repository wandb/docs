# File



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L2737-L2816)



File is a class associated with a file saved by wandb.

```python
File(
 client, attrs
)
```







| Attributes | |
| :--- | :--- |



## Methods

### `delete`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L2796-L2809)

```python
delete()
```




### `display`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L954-L965)

```python
display(
 height=420, hidden=(False)
) -> bool
```

Display this object in jupyter.


### `download`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L2763-L2794)

```python
download(
 root: str = ".",
 replace: bool = (False),
 exist_ok: bool = (False)
) -> io.TextIOWrapper
```

Downloads a file previously saved by a run from the wandb server.


| Arguments | |
| :--- | :--- |
| replace (boolean): If `True`, download will overwrite a local file if it exists. Defaults to `False`. root (str): Local directory to save the file. Defaults to ".". exist_ok (boolean): If `True`, will not raise ValueError if file already exists and will not re-download unless replace=True. Defaults to `False`. |



| Raises | |
| :--- | :--- |
| `ValueError` if file already exists, replace=False and exist_ok=False. |



### `snake_to_camel`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L950-L952)

```python
snake_to_camel(
 string
)
```




### `to_html`



[View source](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L967-L968)

```python
to_html(
 *args, **kwargs
)
```






