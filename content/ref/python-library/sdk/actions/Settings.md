---
title: Settings
object_type: python_sdk_actions
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/sdk/wandb_settings.py >}}




## <kbd>class</kbd> `Settings`
Settings for the W&B SDK. 

This class manages configuration settings for the W&B SDK, ensuring type safety and validation of all settings. Settings are accessible as attributes and can be initialized programmatically, through environment variables (`WANDB_ prefix`), and with configuration files. 

The settings are organized into three categories: 1. Public settings: Core configuration options that users can safely modify to customize  W&B's behavior for their specific needs. 2. Internal settings: Settings prefixed with 'x_' that handle low-level SDK behavior.  These settings are primarily for internal use and debugging. While they can be modified,  they are not considered part of the public API and may change without notice in future  versions. 3. Computed settings: Read-only settings that are automatically derived from other settings or  the environment. 


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

### <kbd>method</kbd> `Settings.to_proto`

```python
to_proto() → wandb_settings_pb2.Settings
```

Generate a protobuf representation of the settings. 

---

### <kbd>method</kbd> `Settings.update_from_dict`

```python
update_from_dict(settings: 'Dict[str, Any]') → None
```

Update settings from a dictionary. 

---

### <kbd>method</kbd> `Settings.update_from_env_vars`

```python
update_from_env_vars(environ: 'Dict[str, Any]')
```

Update settings from environment variables. 

---

### <kbd>method</kbd> `Settings.update_from_settings`

```python
update_from_settings(settings: 'Settings') → None
```

Update settings from another instance of `Settings`. 

---

### <kbd>method</kbd> `Settings.update_from_system_config_file`

```python
update_from_system_config_file()
```

Update settings from the system config file. 

---

### <kbd>method</kbd> `Settings.update_from_system_environment`

```python
update_from_system_environment()
```

Update settings from the system environment. 

---

### <kbd>method</kbd> `Settings.update_from_workspace_config_file`

```python
update_from_workspace_config_file()
```

Update settings from the workspace config file. 

---































