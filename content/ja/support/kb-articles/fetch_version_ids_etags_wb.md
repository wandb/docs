---
title: W&B でこれらのバージョン ID と ETag を取得するにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-fetch_version_ids_etags_wb
support:
- アーティファクト
toc_hide: true
type: docs
url: /support/:filename
---

W&B でアーティファクトリファレンスをログし、かつバケットでバージョン管理が有効になっている場合、Amazon S3 の UI でバージョン ID が表示されます。これらのバージョン ID や ETag を W&B で取得するには、アーティファクトをフェッチして該当するマニフェストエントリにアクセスします。例：

```python
artifact = run.use_artifact("my_table:latest")
for entry in artifact.manifest.entries.values():
    # バージョンIDを取得
    versionID = entry.extra.get("versionID")
    # ETagを取得
    etag = entry.extra.get("etag")
```
