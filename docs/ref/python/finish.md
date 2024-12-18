# finish

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.19.1/wandb/sdk/wandb_run.py#L4045-L4066' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


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
