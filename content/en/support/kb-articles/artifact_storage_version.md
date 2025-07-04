---
url: /support/:filename
title: How much storage does each artifact version use?
toc_hide: true
type: docs
support:
- artifacts
- storage
---
Only files that change between two artifact versions incur storage costs.

{{< img src="/images/artifacts/artifacts-dedupe.PNG" alt="Artifact deduplication" >}}

Consider an image artifact named `animals` that contains two image files, `cat.png` and `dog.png`:

```
images
|-- cat.png (2MB) # Added in `v0`
|-- dog.png (1MB) # Added in `v0`
```

This artifact receives version `v0`.

When adding a new image, `rat.png`, a new artifact version, `v1`, is created with the following contents:

```
images
|-- cat.png (2MB) # Added in `v0`
|-- dog.png (1MB) # Added in `v0`
|-- rat.png (3MB) # Added in `v1`
```

Version `v1` tracks a total of 6MB, but occupies only 3MB of space since it shares the remaining 3MB with `v0`. Deleting `v1` reclaims the 3MB of storage associated with `rat.png`. Deleting `v0` transfers the storage costs of `cat.png` and `dog.png` to `v1`, increasing its storage size to 6MB.
