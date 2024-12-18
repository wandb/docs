---
title: finish
---

{{< cta-button githubLink="https://www.github.com/wandb/wandb/tree/v0.18.7/wandb/sdk/wandb_run.py#L4140-L4151">}}

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
