# finish

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_run.py#L4251-L4262' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

run を終了としてマークし、すべてのデータのアップロードを完了します。

```python
finish(
    exit_code: Optional[int] = None,
    quiet: Optional[bool] = None
) -> None
```

同じプロセスで複数の run を作成する際に使用されます。スクリプトが終了するとき、このメソッドは自動的に呼び出されます。

| 引数 |  |
| :--- | :--- |
|  `exit_code` |  0以外の値を設定して run を失敗としてマークする |
|  `quiet` |  ログ出力を最小限に抑えるために true を設定する |