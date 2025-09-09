---
title: 'InitStartError: wandb プロセスとの通信中にエラーが発生しました'
menu:
  support:
    identifier: ja-support-kb-articles-initstarterror_error_communicating_wandb_process
support:
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

このエラーは、データ を サーバー に同期するプロセスを起動する際に、ライブラリで問題が発生したことを示します。
特定の 環境 でこの問題を解決する回避策は次のとおりです:

{{< tabpane text=true >}}
{{% tab "Linux と OS X" %}}
```python
wandb.init(settings=wandb.Settings(start_method="fork"))
```

{{% /tab %}}
{{% tab "Google Colab" %}}

バージョン `0.13.0` より前では、次を使用してください:

```python
wandb.init(settings=wandb.Settings(start_method="thread"))
```
{{% /tab %}}
{{< /tabpane >}}