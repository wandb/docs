---
title: sweep()
data_type_classification: function
menu:
  reference:
    identifier: ja-ref-python-sdk-functions-sweep
object_type: python_sdk_actions
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_sweep.py >}}




### <kbd>function</kbd> `sweep`

```python
sweep(
    sweep: Union[dict, Callable],
    entity: Optional[str] = None,
    project: Optional[str] = None,
    prior_runs: Optional[List[str]] = None
) → str
```

ハイパーパラメーター探索（スイープ）を初期化します。

機械学習モデルのコスト関数を最適化するために、さまざまなハイパーパラメーターの組み合わせをテストします。

返される一意の識別子 `sweep_id` を控えておいてください。後のステップでエージェントに `sweep_id` を渡します。

自分のスイープをどのように定義するかについては、[Sweep configuration structure](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration) を参照してください。



**引数:**
 
 - `sweep`:  ハイパーパラメーター探索の設定（または設定ジェネレーター）。callable を渡す場合、引数を取らず、W&B の sweep config 仕様に準拠した辞書を返す必要があります。
 - `entity`:  このスイープによって作成される W&B Run を送信する先のユーザー名またはチーム名。指定した Entity が既存であることを確認してください。entity を指定しない場合は、通常あなたのユーザー名となるデフォルトの Entity へ送信されます。
 - `project`:  スイープから作成された W&B Run を送信する Project 名。project を指定しない場合は、「Uncategorized」というプロジェクトに送信されます。
 - `prior_runs`:  このスイープに追加する既存の Run の ID のリスト。



**戻り値:**
 
 - `sweep_id`:  (str) スイープの一意な識別子。
```