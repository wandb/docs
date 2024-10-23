---
title: "How can I fetch these Version IDs and ETags in W&B?"
displayed_sidebar: support
tags:
   - artifacts
---
If an artifact reference is logged with W&B and versioning is enabled on the buckets, the version IDs appear in the S3 UI. To retrieve these version IDs and ETags in W&B, fetch the artifact and access the corresponding manifest entries. For example:

```python
artifact = run.use_artifact("my_table:latest")
for entry in artifact.manifest.entries.values():
    versionID = entry.extra.get("versionID")
    etag = entry.extra.get("etag")
```