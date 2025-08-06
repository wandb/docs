---
title: スイープ
data_type_classification: module
menu:
  reference:
    identifier: ja-ref-python-public-api-sweeps
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/sweeps.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API for Sweeps。

このモジュールは、W&B のハイパーパラメータ最適化 Sweeps とやり取りするためのクラスを提供します。



**例:**
 ```python
from wandb.apis.public import Api

# 指定した sweep を取得
sweep = Api().sweep("entity/project/sweep_id")

# sweep のプロパティへアクセス
print(f"Sweep: {sweep.name}")
print(f"State: {sweep.state}")
print(f"Best Loss: {sweep.best_loss}")

# ベストパフォーマンスの run を取得
best_run = sweep.best_run()
print(f"Best Run: {best_run.name}")
print(f"Metrics: {best_run.summary}")
```

**注意:**

> このモジュールは W&B Public API の一部であり、sweep データへの読み取り専用のアクセスを提供します。sweep の作成や制御を行う場合は、メインの wandb パッケージ内の wandb.sweep() および wandb.agent() 関数を使用してください。

## <kbd>class</kbd> `Sweep`
対象の sweep に紐づく run のセットです。



**属性:**
 
 - `runs` (Runs):  run のリスト
 - `id` (str):  Sweep の ID
 - `project` (str):  sweep が属する Project 名
 - `config` (dict):  sweep configuration を含む辞書
 - `state` (str):  sweep の状態。"Finished"、"Failed"、"Crashed"、"Running" のいずれか
 - `expected_run_count` (int):  sweep の想定実行回数

### <kbd>method</kbd> `Sweep.__init__`

```python
__init__(client, entity, project, sweep_id, attrs=None)
```






---

### <kbd>property</kbd> Sweep.config

この sweep で使用される sweep configuration。

---

### <kbd>property</kbd> Sweep.entity

この sweep に紐づく entity。

---

### <kbd>property</kbd> Sweep.expected_run_count

この sweep で想定される run の数を返します。無限の場合は None を返します。

---

### <kbd>property</kbd> Sweep.name

sweep の名前。

sweep に名前があればそれを返します。なければ sweep ID を返します。

---

### <kbd>property</kbd> Sweep.order

sweep の order key を返します。

---

### <kbd>property</kbd> Sweep.path

project のパスを返します。

このパスは、entity・project 名・sweep ID を含んだリストです。

---

### <kbd>property</kbd> Sweep.url

sweep の URL。

sweep の URL は entity・project・"sweeps" という単語・sweep ID.run_id から生成されます。SaaS ユーザーの場合、`https://wandb.ai/entity/project/sweeps/sweeps_ID` という形式になります。

---

### <kbd>property</kbd> Sweep.username

非推奨です。代わりに `Sweep.entity` を使用してください。



---

### <kbd>method</kbd> `Sweep.best_run`

```python
best_run(order=None)
```

config で定義されたメトリクス、または指定した order でソートした際のベストの run を返します。

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

クラウドバックエンドに対してクエリを実行します。

---


### <kbd>method</kbd> `Sweep.to_html`

```python
to_html(height=420, hidden=False)
```

この sweep を表示する iframe を含む HTML を生成します。
