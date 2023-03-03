# controller



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/1725d84a5bc68d5ecf9aedcbcc447e7e2fb1a1cf/wandb/sdk/wandb_sweep.py#L121-L144)



Public sweep controller constructor.

```python
controller(
 sweep_id_or_config: Optional[Union[str, Dict]] = None,
 entity: Optional[str] = None,
 project: Optional[str] = None
)
```





#### Usage:

```python
import wandb
tuner = wandb.controller(...)
print(tuner.sweep_config)
print(tuner.sweep_id)
tuner.configure_search(...)
tuner.configure_stopping(...)
```
