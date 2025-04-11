---
title: 'InitStartError: wandb プロセスとの通信エラー'
menu:
  support:
    identifier: ja-support-kb-articles-initstarterror_error_communicating_wandb_process
support:
- 実験管理
toc_hide: true
type: docs
url: /support/:filename
---

このエラーは、ライブラリがサーバーにデータを同期するプロセスの起動に問題があることを示しています。

以下の回避策は、特定の環境で問題を解決します。

{{< tabpane text=true >}}
{{% tab "Linux and OS X" %}}
```python
wandb.init(settings=wandb.Settings(start_method="fork"))
```

{{% /tab %}}
{{% tab "Google Colab" %}}

バージョン `0.13.0` より前のものには、次を使用してください：

```python
wandb.init(settings=wandb.Settings(start_method="thread"))
```
{{% /tab %}}
{{< /tabpane >}}