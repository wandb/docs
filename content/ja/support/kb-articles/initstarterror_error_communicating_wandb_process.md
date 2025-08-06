---
title: 'InitStartError: wandb プロセスとの通信エラー'
url: /support/:filename
toc_hide: true
type: docs
support:
- 実験
---

このエラーは、ライブラリがサーバーへデータを同期するプロセスの起動時に問題が発生したことを示しています。

以下の回避策は、特定の環境でこの問題を解決します。

{{< tabpane text=true >}}
{{% tab "LinuxとOS X" %}}
```python
wandb.init(settings=wandb.Settings(start_method="fork"))
```

{{% /tab %}}
{{% tab "Google Colab" %}}

`0.13.0` より前のバージョンの場合は、次の設定を使用してください。

```python
wandb.init(settings=wandb.Settings(start_method="thread"))
```
{{% /tab %}}
{{< /tabpane >}}