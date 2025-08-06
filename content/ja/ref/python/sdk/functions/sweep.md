---
title: sweep()
object_type: python_sdk_actions
data_type_classification: function
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

機械学習モデルのコスト関数を最適化するため、さまざまなハイパーパラメーターの組み合わせを試して最適なものを探索します。

戻り値として一意な識別子 `sweep_id` が返されるのでメモしておきましょう。後のステップでこの `sweep_id` を sweep agent に渡します。

スイープの定義方法については [Sweep configuration structure](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration) を参照してください。



**引数:**
 
 - `sweep`:  ハイパーパラメーター探索の設定（もしくは設定を生成する関数）。callable を指定する場合は、引数を取らず、W&B sweep 設定仕様に準拠した辞書を返すことを確認してください。
 - `entity`:  Sweep により作成される W&B run を送信する先のユーザー名またはチーム名。指定した entity が既に存在していることを確認してください。entity を指定しない場合、run はデフォルト entity（通常はご自身のユーザー名）に送信されます。
 - `project`:  Sweep から作成される W&B run を送信するプロジェクト名。プロジェクトを指定しない場合、run は 'Uncategorized' というプロジェクトに送信されます。
 - `prior_runs`:  この sweep に追加する既存 run の run ID のリスト。



**戻り値:**
 
 - `sweep_id`:  (str) Sweep の一意な識別子。
```