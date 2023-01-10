# Runs



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/597de7d094bdab2fa17d5db396c6bc227b2f62c3/wandb/apis/public.py#L1531-L1641)



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



This is generally used indirectly via the `Api`.runs method



| Class Variables | |
| :--- | :--- |
| `QUERY` | |

