---
title: なぜログしたよりも少ないデータポイントしか表示されないのですか？
menu:
  support:
    identifier: ja-support-kb-articles-seeing_fewer_data_points_logged
support:
- 実験
- メトリクス
toc_hide: true
type: docs
url: /support/:filename
---

`Step` 以外の X 軸でメトリクスを可視化する場合、データポイントが少なく表示されることがあります。メトリクスは同じ `Step` でログされている必要があり、これによって同期が保たれます。同じ `Step` でログされたメトリクスのみが、サンプル間の補間時に使用されます。

**ガイドライン**

メトリクスは 1 回の `log()` 呼び出しにまとめてバンドルしましょう。例えば、次のようにする代わりに：

```python
import wandb
with wandb.init() as run:
    run.log({"Precision": precision})
    ...
    run.log({"Recall": recall})
```

このようにまとめて書くことを推奨します：

```python
import wandb
with wandb.init() as run:
    run.log({"Precision": precision, "Recall": recall})
```

step パラメータを手動で制御したい場合、次のようにコード上でメトリクスを同期してください：

```python
with wandb.init() as run:
    step = 100  # 例: step の値
    # Precision と Recall を同じ step でログする
    run.log({"Precision": precision, "Recall": recall}, step=step)
```

両方の `log()` 呼び出しで `step` の値が同じであることを必ず確認してください。そうすることで、同じ step でメトリクスがログされ、サンプルも一緒になります。`step` の値は各呼び出しごとに単調増加しなければなりません。そうでない場合、`step` の値は無視されます。