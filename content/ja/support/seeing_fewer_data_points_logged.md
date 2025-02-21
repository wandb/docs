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

`Step` 以外の X 軸に対してメトリクスを可視化する場合、表示されるデータポイントが少なくなることが予想されます。メトリクスは、同期を維持するために同じ `Step` で ログ される必要があります。同じ `Step` で ログ されたメトリクスのみが、サンプル間の補間中にサンプリングされます。

**ガイドライン**

メトリクスを1つの `log()` 呼び出しにまとめます。たとえば、次のようにする代わりに：

```python
wandb.log({"Precision": precision})
...
wandb.log({"Recall": recall})
```

以下を使用します：

```python
wandb.log({"Precision": precision, "Recall": recall})
```

step パラメータを手動で制御するには、次のように コード 内でメトリクスを同期します。

```python
wandb.log({"Precision": precision}, step=step)
...
wandb.log({"Recall": recall}, step=step)
```

メトリクスが同じ step で ログ され、一緒にサンプリングされるように、`step` の 値 が両方の `log()` 呼び出しで同じであることを確認してください。`step` の 値 は、各呼び出しで単調に増加する必要があります。そうでない場合、`step` の 値 は無視されます。