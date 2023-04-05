# Html



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/sdk/data_types/html.py#L19-L108)



Wandb class for arbitrary html.

```python
Html(
 data: Union[str, 'TextIO'],
 inject: bool = (True)
) -> None
```





| Arguments | |
| :--- | :--- |
| `data` | (string or io object) HTML to display in wandb |
| `inject` | (boolean) Add a stylesheet to the HTML object. If set to False the HTML will pass through unchanged. |



## Methods

### `inject_head`



[View source](https://www.github.com/wandb/client/tree/9f1a662d681e96387ebf650900aef8f19703b575/wandb/sdk/data_types/html.py#L60-L75)

```python
inject_head() -> None
```






