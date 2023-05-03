# Runs

[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを表示](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/apis/public.py#L1550-L1661)

プロジェクトとオプションのフィルターに関連付けられた、runのイタラブルなコレクションです。

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

通常、`Api`.runs メソッドを間接的に使用しています。

| クラス変数 | |
| :--- | :--- |
| `QUERY` | |