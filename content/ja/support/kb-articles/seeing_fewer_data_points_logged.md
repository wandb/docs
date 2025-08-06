---
title: なぜログしたよりも少ないデータポイントしか表示されないのですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 実験
- メトリクス
---

`Step` 以外の X 軸でメトリクスを可視化する場合、表示されるデータポイントが少なくなることがあります。メトリクスは、同期を保つために同じ `Step` でログされている必要があります。同じ `Step` でログされたメトリクスのみが、サンプル間の補間時にサンプリングされます。

**ガイドライン**

メトリクスは、1 回の `log()` コールでまとめてログするようにしましょう。例として、以下のように書く代わりに:

```python
import wandb
with wandb.init() as run:
    run.log({"Precision": precision})
    ...
    run.log({"Recall": recall})
```

次のようにまとめてログしてください:

```python
import wandb
with wandb.init() as run:
    run.log({"Precision": precision, "Recall": recall})
```

step パラメータを手動で制御したい場合は、次のようにコードでメトリクスを同期させてください:

```python
with wandb.init() as run:
    step = 100  # ステップ値の例
    # 同じ step で Precision と Recall をログする
    run.log({"Precision": precision, "Recall": recall}, step=step)
```

メトリクスを同じ step でログしてサンプルを揃えるために、両方の `log()` コールで `step` の値が同じであることを確認してください。また、`step` の値は呼び出しごとに単調増加する必要があります。そうでない場合、`step` の値は無視されます。