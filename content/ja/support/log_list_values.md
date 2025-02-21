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

これらの例では、[`wandb.log()`]({{< relref path="/ref/python/log/" lang="ja" >}}) を使用して、いくつかの異なる方法で損失を ログ 記録する方法を示します。

{{< tabpane text=true >}}
{{% tab "辞書を使用" %}}
```python
wandb.log({f"losses/loss-{ii}": loss for ii, 
  loss in enumerate(losses)})
```
{{% /tab %}}
{{% tab "ヒストグラムとして" %}}
```python
# 損失をヒストグラムに変換
wandb.log({"losses": wandb.Histogram(losses)})  
```
{{% /tab %}}
{{< /tabpane >}}

詳細については、[ログ 記録に関するドキュメント]({{< relref path="/guides/models/track/log/" lang="ja" >}}) を参照してください。
