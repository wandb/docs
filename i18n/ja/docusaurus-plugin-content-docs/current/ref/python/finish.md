# 終了



[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)GitHubでソースを見る](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/sdk/wandb_run.py#L3705-L3716)
すべてのデータのアップロードが完了したら、runを終了済みとしてマークします。

```python
finish(
 exit_code: Optional[int] = None,
 quiet: Optional[bool] = None
) -> None
```
この機能は、同じプロセスで複数のrunsを作成する場合に使用されます。
スクリプトが終了すると、このメソッドが自動的に呼び出されます。

| 引数 |  |
| :--- | :--- |
| `exit_code` | runを失敗としてマークするには、0以外の値に設定してください |
| `quiet` | ログ出力を最小限に抑えるために、trueに設定してください |