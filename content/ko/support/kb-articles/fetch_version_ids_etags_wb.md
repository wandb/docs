---
title: W&B에서 이러한 버전 ID와 ETag를 어떻게 가져올 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-fetch_version_ids_etags_wb
support:
- 아티팩트
toc_hide: true
type: docs
url: /support/:filename
---

W&B 에서 아티팩트 참조가 로그되고 버킷에 버전 관리가 활성화되어 있다면, Amazon S3 UI 에서 버전 ID 를 확인할 수 있습니다. W&B 에서 이러한 버전 ID 와 ETag 를 가져오려면 아티팩트를 가져와 해당 매니페스트 엔트리에 엑세스하면 됩니다. 예시:

```python
artifact = run.use_artifact("my_table:latest")
for entry in artifact.manifest.entries.values():
    # versionID와 etag 를 가져옵니다
    versionID = entry.extra.get("versionID")
    etag = entry.extra.get("etag")
```
