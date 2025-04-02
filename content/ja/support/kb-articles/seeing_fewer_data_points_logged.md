---
title: Why am I seeing fewer data points than I logged?
menu:
  support:
    identifier: ja-support-kb-articles-seeing_fewer_data_points_logged
support:
- experiments
- metrics
toc_hide: true
type: docs
url: /support/:filename
---

`Step`以外のX軸に対してメトリクスを可視化する場合、表示されるデータ点が少なくなることが予想されます。メトリクスは、同期を維持するために同じ`Step`でログに記録する必要があります。同じ`Step`でログに記録されたメトリクスのみが、サンプル間の補間中にサンプリングされます。

**ガイドライン**

複数のメトリクスを1つの `log()` コールにまとめてください。たとえば、次のようにする代わりに：

```python
wandb.log({"Precision": precision})
...
wandb.log({"Recall": recall})
```

次のようにします：

```python
wandb.log({"Precision": precision, "Recall": recall})
```

step パラメータを手動で制御するには、コード内で次のようにメトリクスを同期させます。

```python
wandb.log({"Precision": precision}, step=step)
...
wandb.log({"Recall": recall}, step=step)
```

メトリクスが同じステップでログに記録され、一緒にサンプリングされるように、両方の `log()` コールで `step` の 値 が同じであることを確認してください。`step` の 値 は、各コールで単調に増加する必要があります。そうでない場合、`step` の 値 は無視されます。
