---
title: Settings
object_type: python_sdk_actions
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/sdk/wandb_settings.py >}}




## <kbd>class</kbd> `Settings`
Settings for the W&B SDK. 

This class manages configuration settings for the W&B SDK, ensuring type safety and validation of all settings. Settings are accessible as attributes and can be initialized programmatically, through environment variables (`WANDB_ prefix`), and with configuration files. 

The settings are organized into three categories: 1. Public settings: Core configuration options that users can safely modify to customize  W&B's behavior for their specific needs. 2. Internal settings: Settings prefixed with 'x_' that handle low-level SDK behavior.  These settings are primarily for internal use and debugging. While they can be modified,  they are not considered part of the public API and may change without notice in future  versions. 3. Computed settings: Read-only settings that are automatically derived from other settings or  the environment. 



**Args:**
 
 - `allow_offline_artifacts` (bool):  Flag to allow table artifacts to be  synced in offline mode. 
 - `allow_val_change` (bool):  Flag to allow modification of `Config` values  after they've been set. 
 - `anonymous` (Optional[Literal["allow", "must", "never"]]):  Controls  anonymous data logging. Possible values are: 
    - "never": requires you to link your W&B account before tracking the run, so you don't accidentally create an anonymous run. 
    - "allow": lets a logged-in user track runs with their account, but lets someone who is running the script without a W&B account see the charts in the UI. 
    - "must": sends the run to an anonymous account instead of to a signed-up user account. 
 - `api_key` (Optional[str]):  The W&B API key. 
 - `azure_account_url_to_access_key` (Optional[Dict[str, str]]):  Mapping of  Azure account URLs to their corresponding access keys for Azure  integration. 
 - `base_url` (str):  The URL of the W&B backend for data synchronization. 
 - `code_dir` (Optional[str]):  Directory containing the code to be  tracked by W&B. 
 - `config_paths` (Optional[Sequence[str]]):  Paths to files to load  configuration from into the `Config` object. 
 - `console` (Literal["auto", "off", "wrap", "redirect", "wrap_raw", "wrap_emu"]):  The  type of console capture to be applied. Possible values are: 
    - "auto" - Automatically selects the console capture method based on the system environment and settings. 
    - "off" - Disables console capture. 
    - "redirect" - Redirects low-level file descriptors for capturing output. 
    - "wrap" - Overrides the write methods of sys.stdout/sys.stderr. Will be mapped to either "wrap_raw" or "wrap_emu" based on the state of the system. 
    - "wrap_raw" - Same as "wrap" but captures raw output directly instead of through an emulator. Derived from the `wrap` setting and should not be set manually. 
    - "wrap_emu" - Same as "wrap" but captures output through an emulator.  Derived from the `wrap` setting and should not be set manually. 
 - `console_multipart` (bool):  Whether to produce multipart console log files. 
 - `credentials_file` (str):  Path to file for writing temporary access tokens. 
 - `disable_code` (bool):  Whether to disable capturing the code. 
 - `disable_git` (bool):  Whether to disable capturing the git state. 
 - `disable_job_creation` (bool):  Whether to disable the creation of a  job artifact for W&B Launch. 
 - `docker` (Optional[str]):  The Docker image used to execute the script. 
 - `email` (Optional[str]):  The email address of the user. 
 - `entity` (Optional[str]):  The W&B entity, such as a user or a team. 
 - `organization` (Optional[str]):  The W&B organization. 
 - `force` (bool):  Whether to pass the `force` flag to `wandb.login()`. 
 - `fork_from` (Optional[RunMoment]):  Specifies a point in a previous  execution of a run to fork from. The point is defined by the  run ID, a metric, and its value. Only the metric '_step' is supported. 
 - `git_commit` (Optional[str]):  The git commit hash to associate with  the run. 
 - `git_remote` (str):  The git remote to associate with the run. 
 - `git_remote_url` (Optional[str]):  The URL of the git remote repository. 
 - `git_root` (Optional[str]):  Root directory of the git repository. 
 - `heartbeat_seconds` (int):  Interval in seconds between heartbeat signals  sent to the W&B servers. 
 - `host` (Optional[str]):  Hostname of the machine running the script. 
 - `http_proxy` (Optional[str]):  Custom proxy servers for http requests to W&B. 
 - `https_proxy` (Optional[str]):  Custom proxy servers for https requests to W&B. 
 - `identity_token_file` (Optional[str]):  Path to file containing an identity token (JWT) for authentication. 
 - `ignore_globs` (Sequence[str]):  Unix glob patterns relative to `files_dir` specifying files to exclude from upload. 
 - `init_timeout` (float):  Time in seconds to wait for the `wandb.init` call to complete before timing out. 
 - `insecure_disable_ssl` (bool):  Whether to disable SSL verification. 
 - `job_name` (Optional[str]):  Name of the Launch job running the script. 
 - `job_source` (Optional[Literal["repo", "artifact", "image"]]):  Source type for Launch. 
 - `label_disable` (bool):  Whether to disable automatic labeling features. 
 - `launch` (bool):  Flag to indicate if the run is being launched through W&B Launch. 
 - `launch_config_path` (Optional[str]):  Path to the launch configuration file. 
 - `login_timeout` (Optional[float]):  Time in seconds to wait for login operations before timing out. 
 - `mode` (Literal["online", "offline", "dryrun", "disabled", "run", "shared"]):  The operating mode for W&B logging and synchronization. 
 - `notebook_name` (Optional[str]):  Name of the notebook if running in a Jupyter-like environment. 
 - `program` (Optional[str]):  Path to the script that created the run, if available. 
 - `program_abspath` (Optional[str]):  The absolute path from the root  repository directory to the script that created the run. Root  repository directory is defined as the directory containing  the .git directory, if it exists. Otherwise, it's the current working directory. 
 - `program_relpath` (Optional[str]):  The relative path to the script that created the run. 
 - `project` (Optional[str]):  The W&B project ID. 
 - `quiet` (bool):  Flag to suppress non-essential output. 
 - `reinit` (Union[Literal["default", "return_previous", "finish_previous", "create_new"], bool]):  What  to do when `wandb.init()` is called while a run is active. Options are 
    - "default": Use "finish_previous" in notebooks and "return_previous" otherwise. 
    - "return_previous": Return the most recently created run that is not yet finished. This does not update `wandb.run`; see the "create_new" option. 
    - "finish_previous": Finish all active runs, then return a new run. 
    - "create_new": Create a new run without modifying other active runs. Does not update `wandb.run` and top-level functions like `wandb.log`. Because of this, some older integrations that rely on the global run will not work. 
 - `relogin` (bool):  Whether to force a new login attempt. 
 - `resume` (Optional[Literal["allow", "must", "never", "auto"]]):  Specifies  the resume behavior for the run. The available options are 
    - "must": Resumes from an existing run with the same ID. If no such run exists, it will result in failure. 
    - "allow": Attempts to resume from an existing run with the same ID. If none is found, a new run will be created. 
    - "never": Always starts a new run. If a run with the same ID already exists, it will result in failure. 
    - "auto": Automatically resumes from the most recent failed run on the same machine. 
 - `resume_from` (Optional[RunMoment]):  Specifies a point in a previous execution of a run to resume from. The point is defined by the run ID, a metric, and its value.  Currently, only the metric '_step' is supported. 
 - `resumed` (bool):  Indication from the server about the state of the run. This is different from resume, a user provided flag. 
 - `root_dir` (str):  The root directory to use as the base for all run-related paths. Used to derive the wandb directory and the run directory. 
 - `run_group` (Optional[str]):  Group identifier for related runs. Used for grouping runs in the UI. 
 - `run_id` (Optional[str]):  The ID of the run. 
 - `run_job_type` (Optional[str]):  Type of job being run (e.g., training, evaluation). 
 - `run_name` (Optional[str]):  Human-readable name for the run. 
 - `run_notes` (Optional[str]):  Additional notes or description for the run. 
 - `run_tags` (Optional[Tuple[str, ...]]):  Tags to associate with the run for organization and filtering. 
 - `sagemaker_disable` (bool):  Flag to disable SageMaker-specific functionality. 
 - `save_code` (Optional[bool]):  Whether to save the code associated with the run. 
 - `settings_system` (Optional[str]):  Path to the system-wide settings file. 
 - `show_colors` (Optional[bool]):  Whether to use colored output in the console. 
 - `show_emoji` (Optional[bool]):  Whether to show emoji in the console output. 
 - `show_errors` (bool):  Whether to display error messages. 
 - `show_info` (bool):  Whether to display informational messages. 
 - `show_warnings` (bool):  Whether to display warning messages. 
 - `silent` (bool):  Flag to suppress all output. 
 - `start_method` (Optional[str]):  Method to use for starting subprocesses. 
 - `strict` (Optional[bool]):  Whether to enable strict mode for validation and error checking. 
 - `summary_timeout` (int):  Time in seconds to wait for summary operations before timing out. 
 - `summary_warnings` (int):  Maximum number of summary warnings to display. 
 - `sweep_id` (Optional[str]):  Identifier of the sweep this run belongs to. 
 - `sweep_param_path` (Optional[str]):  Path to the sweep parameters configuration. 
 - `symlink` (bool):  Whether to use symlinks for run directories. 
 - `sync_tensorboard` (Optional[bool]):  Whether to synchronize TensorBoard logs with W&B. 
 - `table_raise_on_max_row_limit_exceeded` (bool):  Whether to raise an exception when table row limits are exceeded. 
 - `username` (Optional[str]):  Username of the user. 


---

### <kbd>property</kbd> Settings.colab_url

The URL to the Colab notebook, if running in Colab. 

---

### <kbd>property</kbd> Settings.deployment





---

### <kbd>property</kbd> Settings.files_dir

Absolute path to the local directory where the run's files are stored. 

---

### <kbd>property</kbd> Settings.is_local





---

### <kbd>property</kbd> Settings.log_dir

The directory for storing log files. 

---

### <kbd>property</kbd> Settings.log_internal

The path to the file to use for internal logs. 

---

### <kbd>property</kbd> Settings.log_symlink_internal

The path to the symlink to the internal log file of the most recent run. 

---

### <kbd>property</kbd> Settings.log_symlink_user

The path to the symlink to the user-process log file of the most recent run. 

---

### <kbd>property</kbd> Settings.log_user

The path to the file to use for user-process logs. 

---

### <kbd>property</kbd> Settings.model_extra

Get extra fields set during validation. 



**Returns:**
  A dictionary of extra fields, or `None` if `config.extra` is not set to `"allow"`. 

---

### <kbd>property</kbd> Settings.model_fields_set

Returns the set of fields that have been explicitly set on this model instance. 



**Returns:**
  A set of strings representing the fields that have been set,  i.e. that were not filled from defaults. 

---

### <kbd>property</kbd> Settings.project_url

The W&B URL where the project can be viewed. 

---

### <kbd>property</kbd> Settings.resume_fname

The path to the resume file. 

---

### <kbd>property</kbd> Settings.run_mode

The mode of the run. Can be either "run" or "offline-run". 

---

### <kbd>property</kbd> Settings.run_url

The W&B URL where the run can be viewed. 

---

### <kbd>property</kbd> Settings.settings_workspace

The path to the workspace settings file. 

---

### <kbd>property</kbd> Settings.sweep_url

The W&B URL where the sweep can be viewed. 

---

### <kbd>property</kbd> Settings.sync_dir

The directory for storing the run's files. 

---

### <kbd>property</kbd> Settings.sync_file

Path to the append-only binary transaction log file. 

---

### <kbd>property</kbd> Settings.sync_symlink_latest

Path to the symlink to the most recent run's transaction log file. 

---

### <kbd>property</kbd> Settings.timespec

The time specification for the run. 

---

### <kbd>property</kbd> Settings.wandb_dir

Full path to the wandb directory. 



---

### <kbd>classmethod</kbd> `Settings.catch_private_settings`

```python
catch_private_settings(values)
```

Check if a private field is provided and assign to the corresponding public one. 

This is a compatibility layer to handle previous versions of the settings. 

---


### <kbd>method</kbd> `Settings.update_from_dict`

```python
update_from_dict(settings: 'Dict[str, Any]') → None
```

Update settings from a dictionary. 

---




































### <kbd>classmethod</kbd> `Settings.validate_x_stats_coreweave_metadata_base_url`

```python
validate_x_stats_coreweave_metadata_base_url(value)
```





---


