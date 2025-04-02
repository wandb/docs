---
title: sweep
menu:
  reference:
    identifier: ja-ref-python-sweep
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/637bddf198525810add5804059001b1b319d6ad1/wandb/sdk/wandb_sweep.py#L34-L92 >}}

ハイパーパラメーター sweep を初期化します。

```python
sweep(
    sweep: Union[dict, Callable],
    entity: Optional[str] = None,
    project: Optional[str] = None,
    prior_runs: Optional[List[str]] = None
) -> str
```

さまざまな組み合わせをテストすることにより、機械学習モデルのコスト関数を最適化するハイパーパラメーターを検索します。

返される一意の識別子 `sweep_id` に注意してください。後のステップで、`sweep_id` を sweep agent に提供します。

| Args |  |
| :--- | :--- |
| `sweep` | ハイパーパラメーター探索の設定。（または設定ジェネレーター）。sweep の定義方法については、[Sweep configuration structure](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration) を参照してください。callable を指定する場合は、callable が引数を受け取らず、W&B sweep config 仕様に準拠する辞書を返すようにしてください。 |
| `entity` | sweep によって作成された W&B の run を送信する ユーザー名または Team 名。指定する Entity がすでに存在することを確認してください。Entity を指定しない場合、run はデフォルトの Entity（通常はユーザー名）に送信されます。 |
| `project` | sweep から作成された W&B の run の送信先となる Project の名前。Project が指定されていない場合、run は「未分類」というラベルの Project に送信されます。 |
| `prior_runs` | この sweep に追加する既存の run の run ID。 |

| Returns |  |
| :--- | :--- |
| `sweep_id` | str. sweep の一意の識別子。 |
