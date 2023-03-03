# Project



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/1725d84a5bc68d5ecf9aedcbcc447e7e2fb1a1cf/wandb/apis/public.py#L1459-L1541)



A project is a namespace for runs.

```python
Project(
 client, entity, project, attrs
)
```







| Attributes | |
| :--- | :--- |



## Methods

### `artifacts_types`



[View source](https://www.github.com/wandb/client/tree/1725d84a5bc68d5ecf9aedcbcc447e7e2fb1a1cf/wandb/apis/public.py#L1492-L1494)

```python
artifacts_types(
 per_page=50
)
```




### `display`



[View source](https://www.github.com/wandb/client/tree/1725d84a5bc68d5ecf9aedcbcc447e7e2fb1a1cf/wandb/apis/public.py#L971-L982)

```python
display(
 height=420, hidden=(False)
) -> bool
```

Display this object in jupyter


### `snake_to_camel`



[View source](https://www.github.com/wandb/client/tree/1725d84a5bc68d5ecf9aedcbcc447e7e2fb1a1cf/wandb/apis/public.py#L967-L969)

```python
snake_to_camel(
 string
)
```




### `sweeps`



[View source](https://www.github.com/wandb/client/tree/1725d84a5bc68d5ecf9aedcbcc447e7e2fb1a1cf/wandb/apis/public.py#L1496-L1541)

```python
sweeps()
```




### `to_html`



[View source](https://www.github.com/wandb/client/tree/1725d84a5bc68d5ecf9aedcbcc447e7e2fb1a1cf/wandb/apis/public.py#L1476-L1484)

```python
to_html(
 height=420, hidden=(False)
)
```

Generate HTML containing an iframe displaying this project




