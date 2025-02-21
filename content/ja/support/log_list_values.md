---
title: How do I log a list of values?
menu:
  support:
    identifier: ja-support-log_list_values
tags:
- logs
- experiments
toc_hide: true
type: docs
---

これらの例では、異なる方法で損失をログする方法を示しています。[`wandb.log()`]({{< relref path="/ref/python/log/" lang="ja" >}}) を使用しています。

{{< tabpane text=true >}}
{{% tab "Using a dictionary" %}}
```python
wandb.log({f"losses/loss-{ii}": loss for ii, 
  loss in enumerate(losses)})
```
{{% /tab %}}
{{% tab "As a histogram" %}}
```python
# 損失をヒストグラムに変換します
wandb.log({"losses": wandb.Histogram(losses)})  
```
{{% /tab %}}
{{< /tabpane >}}

詳細については、[ログに関するドキュメント]({{< relref path="/guides/models/track/log/" lang="ja" >}}) を参照してください。