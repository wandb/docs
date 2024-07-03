# sweep

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.17.3/wandb/sdk/wandb_sweep.py#L31-L89' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>

ハイパーパラメーター探索を初期化します。

```python
sweep(
    sweep: Union[dict, Callable],
    entity: Optional[str] = None,
    project: Optional[str] = None,
    prior_runs: Optional[List[str]] = None
) -> str
```

機械学習モデルのコスト関数を最適化するようなハイパーパラメーターを、さまざまな組み合わせを試して探索します。

返される一意の識別子 `sweep_id` に注目してください。後のステップで、この `sweep_id` を sweep agent に提供します。

| 引数 |  |
| :--- | :--- |
|  `sweep` | ハイパーパラメーター探索の設定（または設定ジェネレータ）。sweep の定義方法については [Sweep configuration structure](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration) を参照してください。呼び出し可能なものを提供する場合、その呼び出しが引数を取らず、W&B sweep 設定仕様に準拠した辞書を返すことを確認してください。 |
|  `entity` | sweep によって作成された W&B runs を送信したいユーザー名またはチーム名。指定した entity が既に存在することを確認してください。entity を指定しない場合、run はデフォルトの entity（通常はユーザー名）に送信されます。 |
|  `project` | sweep から作成された W&B runs が送信されるプロジェクトの名前。プロジェクトが指定されていない場合、run は「Uncategorized」というラベルの付いたプロジェクトに送信されます。 |
|  `prior_runs` | この sweep に追加する既存の runs の IDs。 |

| 戻り値 |  |
| :--- | :--- |
|  `sweep_id` | str。sweep の一意識別子。 |