---
title: sweep
menu:
  reference:
    identifier: ja-ref-python-sweep
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_sweep.py#L34-L92 >}}

ハイパーパラメーター探索を初期化します。

```python
sweep(
    sweep: Union[dict, Callable],
    entity: Optional[str] = None,
    project: Optional[str] = None,
    prior_runs: Optional[List[str]] = None
) -> str
```

機械学習モデルのコスト関数を最適化するハイパーパラメーターを見つけるために、さまざまな組み合わせをテストします。

返されるユニークな識別子 `sweep_id` をメモしてください。後のステップで `sweep_id` を sweep agent に提供します。

| 引数 |  |
| :--- | :--- |
|  `sweep` |  ハイパーパラメーター探索の設定です。（または設定ジェネレーター）。sweep を定義する方法については、[Sweep configuration structure](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration) を参照してください。コール可能なオブジェクトを提供する場合、引数を取らないことを確認し、W&B sweep config仕様に準拠した辞書を返すようにしてください。|
|  `entity` |  スイープによって作成された W&B run を送信したいユーザー名またはチーム名です。指定した entity が既に存在することを確認してください。もし entity を指定しない場合、run は通常、ユーザー名であるデフォルトの entity に送信されます。 |
|  `project` |  スイープから作成された W&B run が送信されるプロジェクトの名前です。プロジェクトが指定されない場合、run は「Uncategorized」とラベル付けされたプロジェクトに送信されます。 |
|  `prior_runs` |  このスイープに追加する既存の run の ID です。 |

| 戻り値 |  |
| :--- | :--- |
|  `sweep_id` |  str. スイープのためのユニークな識別子です。|