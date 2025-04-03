---
title: How can I fetch these Version IDs and ETags in W&B?
menu:
  support:
    identifier: ja-support-kb-articles-fetch_version_ids_etags_wb
support:
- artifacts
toc_hide: true
type: docs
url: /support/:filename
---

W&B でアーティファクト参照をログに記録し、 バケットで バージョン管理が有効になっている場合、バージョン ID が Amazon S3 UI に表示されます。これらのバージョン ID と ETag を W&B で取得するには、アーティファクトをフェッチし、対応するマニフェストエントリにアクセスします。以下に例を示します。

```python
artifact = run.use_artifact("my_table:latest")
for entry in artifact.manifest.entries.values():
    versionID = entry.extra.get("versionID")
    etag = entry.extra.get("etag")
```
