---
title: 値のリストをログするにはどうすればよいですか？
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

これらの例では、[`wandb.Run.log()`]({{< relref path="/ref/python/sdk/classes/run/#method-runlog/" lang="ja" >}}) を使って損失をいくつかの方法でログする方法を示します。

{{< tabpane text=true >}}
{{% tab "辞書を使う" %}}
```python
import wandb

# 新しい run を初期化
with wandb.init(project="log-list-values", name="log-dict") as run:
    # 損失を辞書としてログする
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
    # 損失をヒストグラムとしてログする
    losses = [0.1, 0.2, 0.3, 0.4, 0.5]
    run.log({"losses": wandb.Histogram(losses)})
```
{{% /tab %}}
{{< /tabpane >}}

詳しくは、[ログに関するドキュメント]({{< relref path="/guides/models/track/log/" lang="ja" >}}) を参照してください。