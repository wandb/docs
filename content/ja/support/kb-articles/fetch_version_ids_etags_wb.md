---
title: W&B で、これらのバージョン ID と ETag を取得するにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-fetch_version_ids_etags_wb
support:
- artifacts
toc_hide: true
type: docs
url: /support/:filename
---

アーティファクト参照が W&B でログされ、バケットでバージョン管理が有効になっている場合、バージョン ID は Amazon S3 の UI に表示されます。これらのバージョン ID と ETag を W&B で取得するには、アーティファクトを取得して、対応するマニフェストのエントリにアクセスします。例えば:

```python
artifact = run.use_artifact("my_table:latest")
for entry in artifact.manifest.entries.values():
    versionID = entry.extra.get("versionID")
    etag = entry.extra.get("etag")
```