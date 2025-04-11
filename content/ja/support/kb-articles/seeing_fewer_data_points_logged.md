---
title: なぜログされたデータポイントが少なく表示されるのですか？
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

メトリクスを `Step` 以外の X 軸に対して視覚化する場合、データポイントが少なくなることがあります。メトリクスは同じ `Step` でログする必要があり、同期を維持します。同じ `Step` でログされるメトリクスのみが、サンプル間の補間中にサンプリングされます。

**ガイドライン**

メトリクスを単一の `log()` 呼び出しにバンドルします。例えば、以下のようにするのではなく:

```python
wandb.log({"Precision": precision})
...
wandb.log({"Recall": recall})
```

以下のようにします:

```python
wandb.log({"Precision": precision, "Recall": recall})
```

ステップパラメータを手動で制御する場合、コード内でメトリクスを次のように同期させます:

```python
wandb.log({"Precision": precision}, step=step)
...
wandb.log({"Recall": recall}, step=step)
```

メトリクスを同じステップでログし、同時にサンプリングするために、両方の `log()` 呼び出しで `step` 値が同じままであることを確認してください。`step` 値は各呼び出しで単調に増加する必要があります。そうでない場合、`step` 値は無視されます。