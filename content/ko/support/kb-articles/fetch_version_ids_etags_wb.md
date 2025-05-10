---
title: How can I fetch these Version IDs and ETags in W&B?
menu:
  support:
    identifier: ko-support-kb-articles-fetch_version_ids_etags_wb
support:
- artifacts
toc_hide: true
type: docs
url: /ko/support/:filename
translationKey: fetch_version_ids_etags_wb
---
만약 아티팩트 참조가 W&B에 기록되고 버킷에서 버전 관리가 활성화되어 있다면, 버전 ID가 Amazon S3 UI에 나타납니다. 이러한 버전 ID와 ETag를 W&B에서 검색하려면, 아티팩트를 가져와서 해당 매니페스트 항목에 엑세스하세요. 예를 들어:

```python
artifact = run.use_artifact("my_table:latest")
for entry in artifact.manifest.entries.values():
    versionID = entry.extra.get("versionID")
    etag = entry.extra.get("etag")
```
