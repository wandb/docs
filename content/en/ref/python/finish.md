---
title: finish
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/sdk/wandb_run.py#L4039-L4060 >}}

Finish a run and upload any remaining data.

```python
finish(
    exit_code: (int | None) = None,
    quiet: (bool | None) = None
) -> None
```

Marks the completion of a W&B run and ensures all data is synced to the server.
The run's final state is determined by its exit conditions and sync status.

#### Run States:

- Running: Active run that is logging data and/or sending heartbeats.
- Crashed: Run that stopped sending heartbeats unexpectedly.
- Finished: Run completed successfully (`exit_code=0`) with all data synced.
- Failed: Run completed with errors (`exit_code!=0`).

| Args |  |
| :--- | :--- |
|  `exit_code` |  Integer indicating the run's exit status. Use 0 for success, any other value marks the run as failed. |
|  `quiet` |  Deprecated. Configure logging verbosity using `wandb.Settings(quiet=...)`. |
