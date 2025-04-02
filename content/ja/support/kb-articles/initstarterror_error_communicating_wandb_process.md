---
title: 'InitStartError: Error communicating with wandb process'
menu:
  support:
    identifier: ja-support-kb-articles-initstarterror_error_communicating_wandb_process
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

このエラーは、ライブラリ が サーバー に データ を同期する プロセス の ローンチ で問題が発生したことを示します。

以下の回避策は、特定の 環境 で問題を解決します。

{{< tabpane text=true >}}
{{% tab "Linux and OS X" %}}
```python
wandb.init(settings=wandb.Settings(start_method="fork"))
```

{{% /tab %}}
{{% tab "Google Colab" %}}

`0.13.0` より前の バージョン の場合は、以下を使用してください。

```python
wandb.init(settings=wandb.Settings(start_method="thread"))
```
{{% /tab %}}
{{< /tabpane >}}
