---
title: 値のリストをログするにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-log_list_values
support:
- ログ
- 実験管理
toc_hide: true
type: docs
url: /ja/support/:filename
---

これらの例は、異なる方法で損失をログとして記録する [`wandb.log()`]({{< relref path="/ref/python/log/" lang="ja" >}}) の使い方を示しています。
{{< tabpane text=true >}}
{{% tab "辞書を使う" %}}
```python
wandb.log({f"losses/loss-{ii}": loss for ii, 
  loss in enumerate(losses)})
```
{{% /tab %}}
{{% tab "ヒストグラムとして" %}}
```python
# 損失をヒストグラムに変換します
wandb.log({"losses": wandb.Histogram(losses)})  
```
{{% /tab %}}
{{< /tabpane >}}

詳細は [ログのドキュメント]({{< relref path="/guides/models/track/log/" lang="ja" >}}) を参照してください。