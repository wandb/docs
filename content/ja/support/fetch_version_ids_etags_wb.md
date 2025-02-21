---
title: How can I fetch these Version IDs and ETags in W&B?
menu:
  support:
    identifier: ja-support-fetch_version_ids_etags_wb
tags:
- artifacts
toc_hide: true
type: docs
---

W&B で artifact reference がログに記録され、バケットで バージョン管理 が有効になっている場合、バージョン ID が Amazon S3 UI に表示されます。これらのバージョン ID と ETag を W&B で取得するには、artifact をフェッチし、対応するマニフェストエントリにアクセスします。例：

```python
artifact = run.use_artifact("my_table:latest")
for entry in artifact.manifest.entries.values():
    versionID = entry.extra.get("versionID")
    etag = entry.extra.get("etag")
```
