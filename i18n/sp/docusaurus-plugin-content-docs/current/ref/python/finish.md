# finish



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/1725d84a5bc68d5ecf9aedcbcc447e7e2fb1a1cf/wandb/sdk/wandb_run.py#L3660-L3671)



Marks a run as finished, and finishes uploading all data.

```python
finish(
 exit_code: Optional[int] = None,
 quiet: Optional[bool] = None
) -> None
```




This is used when creating multiple runs in the same process.
We automatically call this method when your script exits.

| Arguments | |
| :--- | :--- |
| `exit_code` | Set to something other than 0 to mark a run as failed |
| `quiet` | Set to true to minimize log output |

