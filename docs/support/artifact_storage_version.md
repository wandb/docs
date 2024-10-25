---
title: How much storage does each artifact version use?
tags:
- artifact
- storage
---
Only files that change between two artifact versions incur a storage cost.

![v1 of the artifact "dataset" only has 2/5 images that differ, so it only uses 40% of the space.](/images/artifacts/artifacts-dedupe.PNG)

For example, suppose you create an image artifact named `animals` that contains two image files cat.png and dog.png:

```
images
|-- cat.png (2MB) # Added in `v0`
|-- dog.png (1MB) # Added in `v0`
```

This artifact will automatically be assigned a version `v0`.

If you add a new image `rat.png` to your artifact, a new artifact version is create, `v1`, and it will have the following contents:

```
images
|-- cat.png (2MB) # Added in `v0`
|-- dog.png (1MB) # Added in `v0`
|-- rat.png (3MB) # Added in `v1`
```

`v1` tracks a total of 6MB worth of files, however, it only takes up 3MB of space because it shares the remaining 3MB in common with `v0`. If you delete `v1`, you will reclaim the 3MB of storage associated with `rat.png`. If you delete `v0`, then `v1` will inherit the storage costs of `cat.png` and `dog.png` bringing its storage size to 6MB.
