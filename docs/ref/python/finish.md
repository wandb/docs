# finish



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L3705-L3716)



Mark a run as finished, and finish uploading all data.

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

