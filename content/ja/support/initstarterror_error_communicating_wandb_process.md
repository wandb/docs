---
title: 'InitStartError: Error communicating with wandb process'
menu:
  support:
    identifier: ja-support-initstarterror_error_communicating_wandb_process
tags:
- experiments
toc_hide: true
type: docs
---

このエラーは、ライブラリ が サーバー に データ を同期する プロセス の ローンチ で問題が発生したことを示しています。

以下の回避策で、特定の 環境 で問題を解決できます。

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
