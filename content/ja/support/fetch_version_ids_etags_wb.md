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

W&Bでアーティファクト参照がログされ、バケットでバージョン管理が有効になっている場合、バージョンIDはAmazon S3 UIに表示されます。これらのバージョンIDとETagsをW&Bで取得するには、アーティファクトを取得し、対応するマニフェストエントリにアクセスします。例えば：

```python
artifact = run.use_artifact("my_table:latest")
for entry in artifact.manifest.entries.values():
    versionID = entry.extra.get("versionID")
    etag = entry.extra.get("etag")
```