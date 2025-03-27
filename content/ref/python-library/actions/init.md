---
title: init
object_type: api
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/sdk/wandb_init.py >}}




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
    reinit: 'bool | None' = None,
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

`wandb.init()` spawns a new background process to log data to a run, and it also syncs data to https://wandb.ai by default, so you can see your results in real-time. When you're done logging data, call `wandb.finish()` to end the run. If you don't call `wandb.finish()`, the run will end when your script exits. 

Call `wandb.init()` to start a run before logging data with `wandb.log()`: 



**Args:**
 
 - `project`:  The name of the project where you're sending  the new run. If the project is not specified, we will try to infer  the project name from git root or the current program file. If we  can't infer the project name, we will default to `"uncategorized"`. 
 - `entity`:  An entity is a username or team name where  you're sending runs. This entity must exist before you can send runs  there, so make sure to create your account or team in the UI before  starting to log runs. If you don't specify an entity, the run is  sent to your default entity. 
 - `config`:  This sets `wandb.config`,  a dictionary-like object for saving inputs  to your job, like hyperparameters for a model or settings for a data  preprocessing job. The config will show up in a table in the UI that  you can use to group, filter, and sort runs. Keys should not contain  `.` in their names, and values should be under 10 MB.  If `dict`, argparse or `absl.flags` will load the key value pairs into  the `wandb.config` object.  If `str`, will look for a yaml file by that name, and load config from  that file into the `wandb.config` object. 
 - `save_code`:  Turn this on to save the main script or  notebook to W&B. This is valuable for improving experiment  reproducibility and to diff code across experiments in the UI. By  default this is off, but you can flip the default behavior to on  in your account's settings page. 
 - `group`:  Specify a group to organize individual runs into  a larger experiment. For example, you might be doing cross  validation, or you might have multiple jobs that train and evaluate  a model against different test sets. Group gives you a way to  organize runs together into a larger whole, and you can toggle this  on and off in the UI. 
 - `job_type`:  Specify the type of run, which is useful when  you're grouping runs together into larger experiments using group.  For example, you might have multiple jobs in a group, with job types  like train and eval. Setting this makes it easy to filter and group  similar runs together in the UI so you can compare apples to apples. 
 - `tags`:  A list of strings, which will populate the list  of tags on this run in the UI. Tags are useful for organizing runs  together, or applying temporary labels like "baseline" or  "production". It's easy to add and remove tags in the UI, or filter  down to just runs with a specific tag.  If you are resuming a run, its tags will be overwritten by the tags  you pass to `wandb.init()`. If you want to add tags to a resumed run  without overwriting its existing tags, use `run.tags += ["new_tag"]`  after `wandb.init()`. 
 - `name`:  A short display name for this run, which is how  you'll identify this run in the UI. By default, we generate a random  two-word name that lets you easily cross-reference runs from the  table to charts. Keeping these run names short makes the chart  legends and tables easier to read. If you're looking for a place to  save your hyperparameters, we recommend saving those in config. 
 - `notes`:  A longer description of the run, like a `-m` commit  message in git. This helps you remember what you were doing when you  ran this run. 
 - `dir`:  An absolute path to a directory where  metadata will be stored. When you call `download()` on an artifact,  this is the directory where downloaded files will be saved. By default,  this is the `./wandb` directory. 
 - `resume`:  Sets the resuming behavior. Options:  `"allow"`, `"must"`, `"never"`, `"auto"` or `None`. Defaults to `None`. 
    - `None` (default): If the new run has the same ID as a previous run,  this run overwrites that data. 
    - `"auto"` (or `True`): if the previous run on this machine crashed,  automatically resume it. Otherwise, start a new run. 
    - `"allow"`: if id is set with `init(id="UNIQUE_ID")` or  `WANDB_RUN_ID="UNIQUE_ID"` and it is identical to a previous run,  wandb will automatically resume the run with that id. Otherwise,  wandb will start a new run. 
    - `"never"`: if id is set with `init(id="UNIQUE_ID")` or  `WANDB_RUN_ID="UNIQUE_ID"` and it is identical to a previous run,  wandb will crash. 
    - `"must"`: if id is set with `init(id="UNIQUE_ID")` or  `WANDB_RUN_ID="UNIQUE_ID"` and it is identical to a previous run,  wandb will automatically resume the run with the id. Otherwise,  wandb will crash. 
 - `reinit`:  Allow multiple `wandb.init()` calls in the same  process. Defaults to `False`. 
 - `config_exclude_keys`:  string keys to exclude from  `wandb.config`. 
 - `config_include_keys`:  string keys to include in  `wandb.config`. 
 - `anonymous`:  Controls anonymous data logging. 
    - `"never"` By default, you must link your W&B account before  tracking the run, so you don't accidentally create an anonymous  run. 
    - `"allow"`: lets a logged-in user track runs with their account, but  lets someone who is running the script without a W&B account see  the charts in the UI. 
    - `"must"`: sends the run to an anonymous account instead of to a  signed-up user account. 
 - `mode`:  Can be `"online"`, `"offline"` or `"disabled"`. Defaults to  online. 
 - `allow_val_change`:  Whether to allow config values to  change after setting the keys once. By default, we throw an exception  if a config value is overwritten. If you want to track something  like a varying learning rate at multiple times during training, use  `wandb.log()` instead. By default, set to `False` in scripts,  `True` in Jupyter. 
 - `force`:  If `True`, this crashes the script if a user isn't  logged in to W&B. If `False`, this will let the script run in  offline mode if a user isn't logged in to W&B. Default to `False`. 
 - `sync_tensorboard`:  Synchronize wandb logs from tensorboard or  tensorboardX and save the relevant events file. Defaults to `False`. 
 - `tensorboard`:  Alias for `sync_tensorboard`, deprecated. 
 - `monitor_gym`:  Automatically log videos of environment when  using OpenAI Gym. Defaults to `False`. 
 - `id`:  A unique ID for this run, used for resuming. It must  be unique in the project, and if you delete a run you can't reuse  the ID. Use the `name` field for a short descriptive name, or `config`  for saving hyperparameters to compare across runs. The ID cannot  contain the following special characters `/\#?%` or :. 
 - `fork_from`:  A string with the format `{run_id}?_step={step}` describing  a moment in a previous run to fork a new run from. Creates a new  run that picks up logging history from the specified run at the  specified moment. The target run must be in the current project. 
 - `resume_from`:  A string with the format `{run_id}?_step={step}`  describing a moment in a previous run to resume a run from.  This allows users to truncate the history logged to a run at an  intermediate step and resume logging from that step. It uses run  forking under the hood. The target run must be in the current  project. 
 - `settings`:  Settings to use for this run. Defaults to `None`. 



**Raises:**
 
 - `Error`:  if some unknown or internal error happened during the run  initialization. 
 - `AuthenticationError`:  if the user failed to provide valid credentials. 
 - `CommError`:  if there was a problem communicating with the WandB server. 
 - `UsageError`:  if the user provided invalid arguments. 
 - `KeyboardInterrupt`:  if user interrupts the run. 



**Returns:**
 A `Run` object. 





**Examples:**
 ```python
import wandb

wandb.init()
# ... calculate metrics, generate media
wandb.log({"accuracy": 0.9})
``` 

`wandb.init()` returns a run object, and you can also access the run object with `wandb.run`: 

```python
import wandb

run = wandb.init()

assert run is wandb.run
``` 

You can change where the run is logged, just like changing the organization, repository, and branch in git: 

```python
# Set where the run is logged
import wandb

    config = {"lr": 0.01, "batch_size": 32}
    with wandb.init(config=config) as run:
         run.config.update({"architecture": "resnet", "depth": 34})

         # ... your training code here ...
``` 

Pass a dictionary-style object as the `config` keyword argument to add metadata, like hyperparameters, to your run. 

```python
# Add metadata about the run to the config
import wandb

config = {"lr": 3e-4, "batch_size": 32}
config.update({"architecture": "resnet", "depth": 34})
wandb.init(config=config)
``` 
