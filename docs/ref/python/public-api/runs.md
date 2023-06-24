# Runs



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/v0.15.5/wandb/apis/public.py#L1525-L1636)



An iterable collection of runs associated with a project and optional filter.

```python
Runs(
 client: "RetryingClient",
 entity: str,
 project: str,
 filters: Optional[Dict[str, Any]] = None,
 order: Optional[str] = None,
 per_page: int = 50,
 include_sweeps: bool = (True)
)
```




This is generally used indirectly via the `Api`.runs method.



| Class Variables | |
| :--- | :--- |
| `QUERY` | |

