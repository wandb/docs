---
title: リスト形式の値をどのようにログできますか？
url: /support/:filename
toc_hide: true
type: docs
support:
- ログ
- 実験
---

これらの例では、[`wandb.Run.log()`]({{< relref "/ref/python/sdk/classes/run/#method-runlog/" >}}) を使って損失値をログするいくつかの方法を紹介します。

{{< tabpane text=true >}}
{{% tab "辞書を使う場合" %}}
```python
import wandb

# 新しい run を初期化
with wandb.init(project="log-list-values", name="log-dict") as run:
    # 損失値を辞書でログ
    losses = [0.1, 0.2, 0.3, 0.4, 0.5]
    run.log({"losses": losses})
    run.log({f"losses/loss-{ii}": loss for ii, loss in enumerate(losses)})
```
{{% /tab %}}
{{% tab "ヒストグラムとして" %}}
```python
import wandb

# 新しい run を初期化
with wandb.init(project="log-list-values", name="log-hist") as run:
    # 損失値をヒストグラムとしてログ
    losses = [0.1, 0.2, 0.3, 0.4, 0.5]
    run.log({"losses": wandb.Histogram(losses)})
```
{{% /tab %}}
{{< /tabpane >}}

詳細は[ログのドキュメント]({{< relref "/guides/models/track/log/" >}}) を参照してください。