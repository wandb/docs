---
title: Why am I seeing fewer data points than I logged?
menu:
  support:
    identifier: ja-support-seeing_fewer_data_points_logged
tags:
- experiments
- metrics
toc_hide: true
type: docs
---

`Step` 以外の X 軸に対するメトリクスを視覚化する場合、データポイントが少なくなることがあります。メトリクスは、同期を維持するために同じ `Step` でログされる必要があります。同じ `Step` でログされたメトリクスのみがサンプルとして選ばれ、サンプル間の補間が行われます。

**ガイドライン**

メトリクスを 1 つの `log()` 呼び出しにまとめます。例えば、次のようにする代わりに:

```python
wandb.log({"Precision": precision})
...
wandb.log({"Recall": recall})
```

以下のように使用します:

```python
wandb.log({"Precision": precision, "Recall": recall})
```

ステップ パラメータを手動で制御するには、以下のようにコード内でメトリクスを同期させます:

```python
wandb.log({"Precision": precision}, step=step)
...
wandb.log({"Recall": recall}, step=step)
```

メトリクスを同じステップでログして一緒にサンプル化するためには、両方の `log()` 呼び出しで `step` の値が同じであることを確認してください。`step` の値は各呼び出しで単調に増加する必要があります。そうでない場合、`step` の値は無視されます。