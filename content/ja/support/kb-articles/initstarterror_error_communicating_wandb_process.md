---
title: 'InitStartError: wandb プロセスとの通信エラー'
menu:
  support:
    identifier: ja-support-kb-articles-initstarterror_error_communicating_wandb_process
support:
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

このエラーは、ライブラリが データ をサーバーに同期するプロセスのローンンチ時に問題が発生したことを示しています。

以下の回避策は、特定の環境でこの問題を解決します。

{{< tabpane text=true >}}
{{% tab "Linux と OS X" %}}
```python
wandb.init(settings=wandb.Settings(start_method="fork"))
# start_method を "fork" に設定することでエラーを回避します
```
{{% /tab %}}
{{% tab "Google Colab" %}}

バージョン `0.13.0` より前の場合は、以下を使用してください:

```python
wandb.init(settings=wandb.Settings(start_method="thread"))
# start_method を "thread" に設定することでエラーを回避します
```
{{% /tab %}}
{{< /tabpane >}}