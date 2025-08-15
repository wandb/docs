---
title: 値のリストをログするにはどうすればいいですか？
menu:
  support:
    identifier: ja-support-kb-articles-log_list_values
support:
- ログ
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

これらの例では、[`wandb.Run.log()`]({{< relref path="/ref/python/sdk/classes/run/#method-runlog/" lang="ja" >}}) を使って損失値を記録するいくつかの方法を紹介しています。

{{< tabpane text=true >}}
{{% tab "辞書を使う場合" %}}
```python
import wandb

# 新しい run を初期化
with wandb.init(project="log-list-values", name="log-dict") as run:
    # 辞書として損失値をログする
    losses = [0.1, 0.2, 0.3, 0.4, 0.5]
    run.log({"losses": losses})
    run.log({f"losses/loss-{ii}": loss for ii, loss in enumerate(losses)})
```
{{% /tab %}}
{{% tab "ヒストグラムとして記録する場合" %}}
```python
import wandb

# 新しい run を初期化
with wandb.init(project="log-list-values", name="log-hist") as run:
    # ヒストグラムとして損失値をログする
    losses = [0.1, 0.2, 0.3, 0.4, 0.5]
    run.log({"losses": wandb.Histogram(losses)})
```
{{% /tab %}}
{{< /tabpane >}}

詳細については、[ログに関するドキュメント]({{< relref path="/guides/models/track/log/" lang="ja" >}}) をご覧ください。