---
title: "How can I fetch these Version IDs and ETags in W&B?"
tags:
   - artifacts
---

If you've logged an artifact reference with W&B and if the versioning is enabled on your buckets then the version IDs can be seen in the S3 UI. To fetch these version IDs and ETags in W&B, you can fetch the artifact and then get the corresponding manifest entries. For example:

```python
artifact = run.use_artifact("my_table:latest")
for entry in artifact.manifest.entries.values():
    versionID = entry.extra.get("versionID")
    etag = manifest_entry.extra.get("etag")
```