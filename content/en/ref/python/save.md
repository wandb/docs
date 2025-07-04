---
title: save
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/sdk/wandb_run.py#L1989-L2083 >}}

Sync one or more files to W&B.

```python
save(
    glob_str: (str | os.PathLike),
    base_path: (str | os.PathLike | None) = None,
    policy: PolicyName = "live"
) -> (bool | list[str])
```

Relative paths are relative to the current working directory.

A Unix glob, such as "myfiles/*", is expanded at the time `save` is
called regardless of the `policy`. In particular, new files are not
picked up automatically.

A `base_path` may be provided to control the directory structure of
uploaded files. It should be a prefix of `glob_str`, and the directory
structure beneath it is preserved. It's best understood through
examples:

```
wandb.save("these/are/myfiles/*")
# => Saves files in a "these/are/myfiles/" folder in the run.

wandb.save("these/are/myfiles/*", base_path="these")
# => Saves files in an "are/myfiles/" folder in the run.

wandb.save("/User/username/Documents/run123/*.txt")
# => Saves files in a "run123/" folder in the run. See note below.

wandb.save("/User/username/Documents/run123/*.txt", base_path="/User")
# => Saves files in a "username/Documents/run123/" folder in the run.

wandb.save("files/*/saveme.txt")
# => Saves each "saveme.txt" file in an appropriate subdirectory
#    of "files/".
```

Note: when given an absolute path or glob and no `base_path`, one
directory level is preserved as in the example above.

| Args |  |
| :--- | :--- |
|  `glob_str` |  A relative or absolute path or Unix glob. |
|  `base_path` |  A path to use to infer a directory structure; see examples. |
|  `policy` |  One of `live`, `now`, or `end`. * live: upload the file as it changes, overwriting the previous version * now: upload the file once now * end: upload file when the run ends |

| Returns |  |
| :--- | :--- |
|  Paths to the symlinks created for the matched files. For historical reasons, this may return a boolean in legacy code. |
