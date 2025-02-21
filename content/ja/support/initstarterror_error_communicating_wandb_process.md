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

このエラーは、ライブラリがデータをサーバーに同期するプロセスをローンチする際に問題が発生したことを示しています。

以下の回避策は、特定の環境での問題を解決します：

{{< tabpane text=true >}}
{{% tab "Linux and OS X" %}}
```python
wandb.init(settings=wandb.Settings(start_method="fork"))
```

{{% /tab %}}
{{% tab "Google Colab" %}}

バージョン `0.13.0` より前のバージョンを使用する場合：

```python
wandb.init(settings=wandb.Settings(start_method="thread"))
```
{{% /tab %}}
{{< /tabpane >}}