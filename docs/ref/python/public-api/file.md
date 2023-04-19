# File



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L2766-L2845)



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



[View source](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L2825-L2838)

```python
delete()
```




### `display`



[View source](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L979-L990)

```python
display(
 height=420, hidden=(False)
) -> bool
```

Display this object in jupyter.


### `download`



[View source](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L2792-L2823)

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



[View source](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L975-L977)

```python
snake_to_camel(
 string
)
```




### `to_html`



[View source](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L992-L993)

```python
to_html(
 *args, **kwargs
)
```






