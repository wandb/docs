---
title: W&B でこれらのバージョン ID と ETag を取得するにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-fetch_version_ids_etags_wb
support:
  - artifacts
toc_hide: true
type: docs
url: /ja/support/:filename
translationKey: fetch_version_ids_etags_wb
---
W&Bでアーティファクト参照がログされ、バケットでバージョン管理が有効になっている場合、バージョンIDがAmazon S3 UIに表示されます。これらのバージョンIDとETagsをW&Bで取得するには、アーティファクトを取得し、対応するマニフェストエントリにアクセスします。例えば:

```python
artifact = run.use_artifact("my_table:latest")
for entry in artifact.manifest.entries.values():
    versionID = entry.extra.get("versionID")
    etag = entry.extra.get("etag")
```
