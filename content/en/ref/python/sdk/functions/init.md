---
title: init()
object_type: python_sdk_actions
data_type_classification: function
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_init.py >}}




### <kbd>function</kbd> `init`

```python
init(
    entity: 'str | None' = None,
    project: 'str | None' = None,
    dir: 'StrPath | None' = None,
    id: 'str | None' = None,
    name: 'str | None' = None,
    notes: 'str | None' = None,
    tags: 'Sequence[str] | None' = None,
    config: 'dict[str, Any] | str | None' = None,
    config_exclude_keys: 'list[str] | None' = None,
    config_include_keys: 'list[str] | None' = None,
    allow_val_change: 'bool | None' = None,
    group: 'str | None' = None,
    job_type: 'str | None' = None,
    mode: "Literal['online', 'offline', 'disabled'] | None" = None,
    force: 'bool | None' = None,
    anonymous: "Literal['never', 'allow', 'must'] | None" = None,
    reinit: "bool | Literal[None, 'default', 'return_previous', 'finish_previous', 'create_new']" = None,
    resume: "bool | Literal['allow', 'never', 'must', 'auto'] | None" = None,
    resume_from: 'str | None' = None,
    fork_from: 'str | None' = None,
    save_code: 'bool | None' = None,
    tensorboard: 'bool | None' = None,
    sync_tensorboard: 'bool | None' = None,
    monitor_gym: 'bool | None' = None,
    settings: 'Settings | dict[str, Any] | None' = None
) → Run
```

Start a new run to track and log to W&B. 

In an ML training pipeline, you could add `wandb.init()` to the beginning of your training script as well as your evaluation script, and each piece would be tracked as a run in W&B. 

`wandb.init()` spawns a new background process to log data to a run, and it also syncs data to https://wandb.ai by default, so you can see your results in real-time. When you're done logging data, call `wandb.finish()` to end the run. If you don't call `run.finish()`, the run will end when your script exits. 

Run IDs must not contain any of the following special characters `/ \ # ? % :` 



**Args:**
 
 - `entity`:  The username or team name the runs are logged to.  The entity must already exist, so ensure you create your account  or team in the UI before starting to log runs. If not specified, the  run will default your default entity. To change the default entity,  go to your settings and update the  "Default location to create new projects" under "Default team". 
 - `project`:  The name of the project under which this run will be logged.  If not specified, we use a heuristic to infer the project name based  on the system, such as checking the git root or the current program  file. If we can't infer the project name, the project will default to  `"uncategorized"`. 
 - `dir`:  The absolute path to the directory where experiment logs and  metadata files are stored. If not specified, this defaults  to the `./wandb` directory. Note that this does not affect the  location where artifacts are stored when calling `download()`. 
 - `id`:  A unique identifier for this run, used for resuming. It must be unique  within the project and cannot be reused once a run is deleted. For  a short descriptive name, use the `name` field,  or for saving hyperparameters to compare across runs, use `config`. 
 - `name`:  A short display name for this run, which appears in the UI to help  you identify it. By default, we generate a random two-word name  allowing easy cross-reference runs from table to charts. Keeping these  run names brief enhances readability in chart legends and tables. For  saving hyperparameters, we recommend using the `config` field. 
 - `notes`:  A detailed description of the run, similar to a commit message in  Git. Use this argument to capture any context or details that may  help you recall the purpose or setup of this run in the future. 
 - `tags`:  A list of tags to label this run in the UI. Tags are helpful for  organizing runs or adding temporary identifiers like "baseline" or  "production." You can easily add, remove tags, or filter by tags in  the UI.  If resuming a run, the tags provided here will replace any existing  tags. To add tags to a resumed run without overwriting the current  tags, use `run.tags += ("new_tag",)` after calling `run = wandb.init()`. 
 - `config`:  Sets `wandb.config`, a dictionary-like object for storing input  parameters to your run, such as model hyperparameters or data  preprocessing settings.  The config appears in the UI in an overview page, allowing you to  group, filter, and sort runs based on these parameters.  Keys should not contain periods (`.`), and values should be  smaller than 10 MB.  If a dictionary, `argparse.Namespace`, or `absl.flags.FLAGS` is  provided, the key-value pairs will be loaded directly into  `wandb.config`.  If a string is provided, it is interpreted as a path to a YAML file,  from which configuration values will be loaded into `wandb.config`. 
 - `config_exclude_keys`:  A list of specific keys to exclude from `wandb.config`. 
 - `config_include_keys`:  A list of specific keys to include in `wandb.config`. 
 - `allow_val_change`:  Controls whether config values can be modified after their  initial set. By default, an exception is raised if a config value is  overwritten. For tracking variables that change during training, such as  a learning rate, consider using `wandb.log()` instead. By default, this  is `False` in scripts and `True` in Notebook environments. 
 - `group`:  Specify a group name to organize individual runs as part of a larger  experiment. This is useful for cases like cross-validation or running  multiple jobs that train and evaluate a model on different test sets.  Grouping allows you to manage related runs collectively in the UI,  making it easy to toggle and review results as a unified experiment. 
 - `job_type`:  Specify the type of run, especially helpful when organizing runs  within a group as part of a larger experiment. For example, in a group,  you might label runs with job types such as "train" and "eval".  Defining job types enables you to easily filter and group similar runs  in the UI, facilitating direct comparisons. 
 - `mode`:  Specifies how run data is managed, with the following options: 
    - `"online"` (default): Enables live syncing with W&B when a network  connection is available, with real-time updates to visualizations. 
    - `"offline"`: Suitable for air-gapped or offline environments; data  is saved locally and can be synced later. Ensure the run folder  is preserved to enable future syncing. 
    - `"disabled"`: Disables all W&B functionality, making the run’s methods  no-ops. Typically used in testing to bypass W&B operations. 
 - `force`:  Determines if a W&B login is required to run the script. If `True`,  the user must be logged in to W&B; otherwise, the script will not  proceed. If `False` (default), the script can proceed without a login,  switching to offline mode if the user is not logged in. 
 - `anonymous`:  Specifies the level of control over anonymous data logging.  Available options are: 
    - `"never"` (default): Requires you to link your W&B account before  tracking the run. This prevents unintentional creation of anonymous  runs by ensuring each run is associated with an account. 
    - `"allow"`: Enables a logged-in user to track runs with their account,  but also allows someone running the script without a W&B account  to view the charts and data in the UI. 
    - `"must"`: Forces the run to be logged to an anonymous account, even  if the user is logged in. 
 - `reinit`:  Shorthand for the "reinit" setting. Determines the behavior of  `wandb.init()` when a run is active. 
 - `resume`:  Controls the behavior when resuming a run with the specified `id`.  Available options are: 
    - `"allow"`: If a run with the specified `id` exists, it will resume  from the last step; otherwise, a new run will be created. 
    - `"never"`: If a run with the specified `id` exists, an error will  be raised. If no such run is found, a new run will be created. 
    - `"must"`: If a run with the specified `id` exists, it will resume  from the last step. If no run is found, an error will be raised. 
    - `"auto"`: Automatically resumes the previous run if it crashed on  this machine; otherwise, starts a new run. 
    - `True`: Deprecated. Use `"auto"` instead. 
    - `False`: Deprecated. Use the default behavior (leaving `resume`  unset) to always start a new run.  If `resume` is set, `fork_from` and `resume_from` cannot be  used. When `resume` is unset, the system will always start a new run. 
 - `resume_from`:  Specifies a moment in a previous run to resume a run from,  using the format `{run_id}?_step={step}`. This allows users to truncate  the history logged to a run at an intermediate step and resume logging  from that step. The target run must be in the same project.  If an `id` argument is also provided, the `resume_from` argument will  take precedence.  `resume`, `resume_from` and `fork_from` cannot be used together, only  one of them can be used at a time.  Note that this feature is in beta and may change in the future. 
 - `fork_from`:  Specifies a point in a previous run from which to fork a new  run, using the format `{id}?_step={step}`. This creates a new run that  resumes logging from the specified step in the target run’s history.  The target run must be part of the current project.  If an `id` argument is also provided, it must be different from the  `fork_from` argument, an error will be raised if they are the same.  `resume`, `resume_from` and `fork_from` cannot be used together, only  one of them can be used at a time.  Note that this feature is in beta and may change in the future. 
 - `save_code`:  Enables saving the main script or notebook to W&B, aiding in  experiment reproducibility and allowing code comparisons across runs in  the UI. By default, this is disabled, but you can change the default to  enable on your settings page. 
 - `tensorboard`:  Deprecated. Use `sync_tensorboard` instead. 
 - `sync_tensorboard`:  Enables automatic syncing of W&B logs from TensorBoard  or TensorBoardX, saving relevant event files for viewing in the W&B UI. 
 - `saving relevant event files for viewing in the W&B UI. (Default`:  `False`) 
 - `monitor_gym`:  Enables automatic logging of videos of the environment when  using OpenAI Gym. 
 - `settings`:  Specifies a dictionary or `wandb.Settings` object with advanced  settings for the run. 



**Raises:**
 
 - `Error`:  If some unknown or internal error happened during the run  initialization. 
 - `AuthenticationError`:  If the user failed to provide valid credentials. 
 - `CommError`:  If there was a problem communicating with the WandB server. 
 - `UsageError`:  If the user provided invalid arguments. 
 - `KeyboardInterrupt`:  If user interrupts the run. 



**Returns:**
 A `Run` object. 





**Examples:**
 `wandb.init()` returns a `Run` object. Use the run object to log data, save artifacts, and manage the run lifecycle. 

```python
import wandb

config = {"lr": 0.01, "batch_size": 32}
with wandb.init(config=config) as run:
    # Log accuracy and loss to the run
    acc = 0.95  # Example accuracy
    loss = 0.05  # Example loss
    run.log({"accuracy": acc, "loss": loss})
``` 
