---
title: スイープ
object_type: public_apis_namespace
data_type_classification: module
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/sweeps.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API for Sweeps

このモジュールは、W&B のハイパーパラメーター最適化 Sweeps とやりとりするためのクラスを提供します。



**例:**
 ```python
from wandb.apis.public import Api

# 特定の sweep を取得
sweep = Api().sweep("entity/project/sweep_id")

# sweep のプロパティにアクセス
print(f"Sweep: {sweep.name}")
print(f"State: {sweep.state}")
print(f"Best Loss: {sweep.best_loss}")

# 最良の run を取得
best_run = sweep.best_run()
print(f"Best Run: {best_run.name}")
print(f"Metrics: {best_run.summary}")
``` 



**注意:**

> このモジュールは W&B Public API の一部であり、sweep のデータを読み込み専用で取得できます。sweep の作成や操作を行いたい場合は、メインの wandb パッケージ内の wandb.sweep() や wandb.agent() メソッドを使用してください。

## <kbd>class</kbd> `Sweep`
sweep に関連付けられている run のセットです。



**属性:**
 
 - `runs` (Runs):  run のリスト
 - `id` (str):  Sweep の ID
 - `project` (str):  sweep が属する Project の名前
 - `config` (dict):  sweep configuration を含む辞書
 - `state` (str):  sweep の状態。"Finished", "Failed", "Crashed", "Running" のいずれか
 - `expected_run_count` (int):  sweep に対して想定されている run の数

### <kbd>method</kbd> `Sweep.__init__`

```python
__init__(client, entity, project, sweep_id, attrs=None)
```






---

### <kbd>property</kbd> Sweep.config

sweep に使用された sweep configuration。

---

### <kbd>property</kbd> Sweep.entity

その sweep に紐づく entity。

---

### <kbd>property</kbd> Sweep.expected_run_count

sweep 内で想定される run の数を返します。無制限の場合は None になります。

---

### <kbd>property</kbd> Sweep.name

sweep の名前。

sweep に名前が設定されていればそれを返し、なければ sweep ID を返します。

---

### <kbd>property</kbd> Sweep.order

sweep の order キーを返します。

---

### <kbd>property</kbd> Sweep.path

Project のパスを返します。

パスは、entity、Project 名、sweep ID で構成されるリストです。

---

### <kbd>property</kbd> Sweep.url

sweep の URL。

この URL は entity、Project、「sweeps」、sweep ID.run_id から作成されます。SaaS ユーザーの場合は `https://wandb.ai/entity/project/sweeps/sweeps_ID` の形式になります。

---

### <kbd>property</kbd> Sweep.username

非推奨。代わりに `Sweep.entity` を利用してください。



---

### <kbd>method</kbd> `Sweep.best_run`

```python
best_run(order=None)
```

config に定義されたメトリクス、または指定した順序で並べ替えて最も良い run を返します。

---

### <kbd>classmethod</kbd> `Sweep.get`

```python
get(
    client,
    entity=None,
    project=None,
    sid=None,
    order=None,
    query=None,
    **kwargs
)
```

クラウドのバックエンドに対してクエリを実行します。

---


### <kbd>method</kbd> `Sweep.to_html`

```python
to_html(height=420, hidden=False)
```

この sweep を表示する iframe を含んだ HTML を生成します。