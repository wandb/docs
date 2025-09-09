---
title: sweep()
data_type_classification: function
menu:
  reference:
    identifier: ja-ref-python-sdk-functions-sweep
object_type: python_sdk_actions
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_sweep.py >}}




### <kbd>関数</kbd> `sweep`

```python
sweep(
    sweep: Union[dict, Callable],
    entity: Optional[str] = None,
    project: Optional[str] = None,
    prior_runs: Optional[List[str]] = None
) → str
```

ハイパーパラメーター探索を初期化します。

さまざまな組み合わせを試し、機械学習 モデルのコスト関数を最適化するハイパーパラメーターを探索します。

返される一意の識別子 `sweep_id` を控えておいてください。後のステップで、その `sweep_id` を sweep agent に渡します。

sweep の定義方法については、[Sweep configuration structure](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration) を参照してください。



**Args:**
 
 - `sweep`:  ハイパーパラメーター探索の設定（または設定ジェネレーター）。呼び出し可能オブジェクトを渡す場合は、引数を取らず、W&B の sweep config 仕様に準拠した 辞書 を返すことを確認してください。 
 - `entity`:  この sweep によって作成される W&B の Runs を送信したいユーザー名または Team 名。指定する entity が既に存在していることを確認してください。entity を指定しない場合、run は通常ユーザー名であるあなたのデフォルトの entity に送信されます。 
 - `project`:  この sweep によって作成された W&B の Runs を送信する Project 名。Project が指定されていない場合、run は 'Uncategorized' というラベルの Project に送信されます。 
 - `prior_runs`:  この sweep に追加する既存の run の ID。 



**Returns:**
 
 - `str`:  sweep の一意な識別子。