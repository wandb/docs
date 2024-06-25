
# sweep

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.1/wandb/sdk/wandb_sweep.py#L31-L87' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

ハイパーパラメータ探索を初期化します。

```python
sweep(
    sweep: Union[dict, Callable],
    entity: Optional[str] = None,
    project: Optional[str] = None
) -> str
```

機械学習モデルのコスト関数を最適化するハイパーパラメータを、さまざまな組み合わせを試して探索します。

一意の識別子 `sweep_id` が返されることに注意してください。後のステップで、`sweep_id` を sweep agent に提供します。

| 引数 |  |
| :--- | :--- |
|  `sweep` |  ハイパーパラメーター探索の設定（または設定ジェネレーター）。sweep の定義方法については、[Sweep configuration structure](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration) を参照してください。コール可能なオブジェクトを提供する場合、そのオブジェクトが引数を取らず、W&B の sweep 設定仕様に準拠する辞書を返すことを確認してください。 |
|  `entity` |  sweep によって作成された W&B run を送信したいユーザー名またはチーム名。指定した entity が既に存在することを確認してください。entity を指定しない場合、run は通常ユーザー名であるデフォルトの entity に送信されます。 |
|  `project` |  sweep により作成された W&B run が送信されるプロジェクトの名前。プロジェクトが指定されていない場合、run は 'Uncategorized' とラベル付けされたプロジェクトに送信されます。 |

| 戻り値 |  |
| :--- | :--- |
|  `sweep_id` |  str。sweep の一意の識別子。 |