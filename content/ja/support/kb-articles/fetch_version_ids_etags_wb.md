---
title: W&B でこれらのバージョン ID と ETag を取得するにはどうすればいいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- アーティファクト
---

W&B でアーティファクト参照がログされ、バケットでバージョン管理が有効になっている場合、 Amazon S3 の UI にバージョン ID が表示されます。これらのバージョン ID と ETag を W&B で取得するには、アーティファクトを取得して対応するマニフェストエントリにアクセスします。例：

```python
artifact = run.use_artifact("my_table:latest")
for entry in artifact.manifest.entries.values():
    # バージョンIDを取得
    versionID = entry.extra.get("versionID")
    # ETag を取得
    etag = entry.extra.get("etag")
```