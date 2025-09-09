---
title: ログした数より少ないデータポイントしか表示されないのはなぜですか？
menu:
  support:
    identifier: ja-support-kb-articles-seeing_fewer_data_points_logged
support:
- 実験管理
- メトリクス
toc_hide: true
type: docs
url: /support/:filename
---

`Step` 以外の X 軸でメトリクスを可視化すると、表示されるデータポイントが少なくなる場合があります。メトリクスは同期を保つため、同じ `Step` でログする必要があります。サンプル間を補間する際には、同じ `Step` でログされたメトリクスだけがサンプリングされます。

**ガイドライン**

メトリクスは 1 回の `log()` 呼び出しにまとめてください。例えば、次のようにする代わりに:

```python
import wandb
with wandb.init() as run:
    run.log({"Precision": precision})
    ...
    run.log({"Recall": recall})
```

次のようにします:

```python
import wandb
with wandb.init() as run:
    run.log({"Precision": precision, "Recall": recall})
```

step パラメータを手動で制御する場合は、次のようにコード内でメトリクスを同期させてください:

```python
with wandb.init() as run:
    step = 100  # step の例となる値
    # Precision と Recall を同じ step でログする
    run.log({"Precision": precision, "Recall": recall}, step=step)
```

メトリクスを同じ step にまとめてログし、同時にサンプリングさせるには、両方の `log()` 呼び出しで `step` の値が同じであることを確認してください。`step` の値は各呼び出しで単調増加する必要があります。そうでない場合、`step` の値は無視されます。