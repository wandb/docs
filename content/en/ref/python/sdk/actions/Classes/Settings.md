---
title: Settings
object_type: python_sdk_actions
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py >}}



Settings for the W&B SDK.

This class manages configuration settings for the W&B SDK,
ensuring type safety and validation of all settings. Settings are accessible
as attributes and can be initialized programmatically, through environment
variables (`WANDB_ prefix`), and with configuration files.

The settings are organized into three categories:
1. Public settings: Core configuration options that users can safely modify to customize
   W&B's behavior for their specific needs.
2. Internal settings: Settings prefixed with 'x_' that handle low-level SDK behavior.
   These settings are primarily for internal use and debugging. While they can be modified,
   they are not considered part of the public API and may change without notice in future
   versions.
3. Computed settings: Read-only settings that are automatically derived from other settings or
   the environment.

Attributes:
- allow_offline_artifacts (bool): Flag to allow table artifacts to be synced in offline mode.
    To revert to the old behavior, set this to False.
- allow_val_change (bool): Flag to allow modification of `Config` values after they've been set.
- anonymous (Optional): Controls anonymous data logging.
    Possible values are:
    - "never": requires you to link your W&B account before
    tracking the run, so you don't accidentally create an anonymous
    run.
    - "allow": lets a logged-in user track runs with their account, but
    lets someone who is running the script without a W&B account see
    the charts in the UI.
    - "must": sends the run to an anonymous account instead of to a
    signed-up user account.
- api_key (Optional): The W&B API key.
- azure_account_url_to_access_key (Optional): Mapping of Azure account URLs to their corresponding access keys for Azure integration.
- base_url (str): The URL of the W&B backend for data synchronization.
- code_dir (Optional): Directory containing the code to be tracked by W&B.
- config_paths (Optional): Paths to files to load configuration from into the `Config` object.
- console (Literal): The type of console capture to be applied.
    Possible values are:
    "auto" - Automatically selects the console capture method based on the
    system environment and settings.
    "off" - Disables console capture.
    "redirect" - Redirects low-level file descriptors for capturing output.
    "wrap" - Overrides the write methods of sys.stdout/sys.stderr. Will be
    mapped to either "wrap_raw" or "wrap_emu" based on the state of the system.
    "wrap_raw" - Same as "wrap" but captures raw output directly instead of
    through an emulator. Derived from the `wrap` setting and should not be set manually.
    "wrap_emu" - Same as "wrap" but captures output through an emulator.
    Derived from the `wrap` setting and should not be set manually.
- console_multipart (bool): Whether to produce multipart console log files.
- credentials_file (str): Path to file for writing temporary access tokens.
- disable_code (bool): Whether to disable capturing the code.
- disable_git (bool): Whether to disable capturing the git state.
- disable_job_creation (bool): Whether to disable the creation of a job artifact for W&B Launch.
- docker (Optional): The Docker image used to execute the script.
- email (Optional): The email address of the user.
- entity (Optional): The W&B entity, such as a user or a team.
- force (bool): Whether to pass the `force` flag to `wandb.login()`.
- fork_from (Optional): Specifies a point in a previous execution of a run to fork from.
    The point is defined by the run ID, a metric, and its value.
    Currently, only the metric '_step' is supported.
- git_commit (Optional): The git commit hash to associate with the run.
- git_remote (str): The git remote to associate with the run.
- git_remote_url (Optional): The URL of the git remote repository.
- git_root (Optional): Root directory of the git repository.
- heartbeat_seconds (int): Interval in seconds between heartbeat signals sent to the W&B servers.
- host (Optional): Hostname of the machine running the script.
- http_proxy (Optional): Custom proxy servers for http requests to W&B.
- https_proxy (Optional): Custom proxy servers for https requests to W&B.
- identity_token_file (Optional): Path to file containing an identity token (JWT) for authentication.
- ignore_globs (Sequence): Unix glob patterns relative to `files_dir` specifying files to exclude from upload.
- init_timeout (float): Time in seconds to wait for the `wandb.init` call to complete before timing out.
- insecure_disable_ssl (bool): Whether to insecurely disable SSL verification.
- job_name (Optional): Name of the Launch job running the script.
- job_source (Optional): Source type for Launch.
- label_disable (bool): Whether to disable automatic labeling features.
- launch (bool): Flag to indicate if the run is being launched through W&B Launch.
- launch_config_path (Optional): Path to the launch configuration file.
- login_timeout (Optional): Time in seconds to wait for login operations before timing out.
- mode (Literal): The operating mode for W&B logging and synchronization.
- notebook_name (Optional): Name of the notebook if running in a Jupyter-like environment.
- organization (Optional): The W&B organization.
- program (Optional): Path to the script that created the run, if available.
- program_abspath (Optional): The absolute path from the root repository directory to the script that
    created the run.
    Root repository directory is defined as the directory containing the
    .git directory, if it exists. Otherwise, it's the current working directory.
- program_relpath (Optional): The relative path to the script that created the run.
- project (Optional): The W&B project ID.
- quiet (bool): Flag to suppress non-essential output.
- reinit (Union): What to do when `wandb.init()` is called while a run is active.
    Options:
    - "default": Use "finish_previous" in notebooks and "return_previous"
    otherwise.
    - "return_previous": Return the most recently created run
    that is not yet finished. This does not update `wandb.run`; see
    the "create_new" option.
    - "finish_previous": Finish all active runs, then return a new run.
    - "create_new": Create a new run without modifying other active runs.
    Does not update `wandb.run` and top-level functions like `wandb.log`.
    Because of this, some older integrations that rely on the global run
    will not work.
    Can also be a boolean, but this is deprecated. False is the same as
    "return_previous", and True is the same as "finish_previous".
- relogin (bool): Flag to force a new login attempt.
- resume (Optional): Specifies the resume behavior for the run.
    The available options are:
    "must": Resumes from an existing run with the same ID. If no such run exists,
    it will result in failure.
    "allow": Attempts to resume from an existing run with the same ID. If none is
    found, a new run will be created.
    "never": Always starts a new run. If a run with the same ID already exists,
    it will result in failure.
    "auto": Automatically resumes from the most recent failed run on the same
    machine.
- resume_from (Optional): Specifies a point in a previous execution of a run to resume from.
    The point is defined by the run ID, a metric, and its value.
    Currently, only the metric '_step' is supported.
- resumed (bool): Indication from the server about the state of the run.
    This is different from resume, a user provided flag.
- root_dir (str): The root directory to use as the base for all run-related paths.
    In particular, this is used to derive the wandb directory and the run directory.
- run_group (Optional): Group identifier for related runs.
    Used for grouping runs in the UI.
- run_id (Optional): The ID of the run.
- run_job_type (Optional): Type of job being run (e.g., training, evaluation).
- run_name (Optional): Human-readable name for the run.
- run_notes (Optional): Additional notes or description for the run.
- run_tags (Optional): Tags to associate with the run for organization and filtering.
- sagemaker_disable (bool): Flag to disable SageMaker-specific functionality.
- save_code (Optional): Whether to save the code associated with the run.
- settings_system (Optional): Path to the system-wide settings file.
- show_colors (Optional): Whether to use colored output in the console.
- show_emoji (Optional): Whether to show emoji in the console output.
- show_errors (bool): Whether to display error messages.
- show_info (bool): Whether to display informational messages.
- show_warnings (bool): Whether to display warning messages.
- silent (bool): Flag to suppress all output.
- start_method (Optional): Method to use for starting subprocesses.
- strict (Optional): Whether to enable strict mode for validation and error checking.
- summary_timeout (int): Time in seconds to wait for summary operations before timing out.
- summary_warnings (int): Maximum number of summary warnings to display.
- sweep_id (Optional): Identifier of the sweep this run belongs to.
- sweep_param_path (Optional): Path to the sweep parameters configuration.
- symlink (bool): Whether to use symlinks (True by default except on Windows).
- sync_tensorboard (Optional): Whether to synchronize TensorBoard logs with W&B.
- table_raise_on_max_row_limit_exceeded (bool): Whether to raise an exception when table row limits are exceeded.
- username (Optional): Username.
- x_cli_only_mode (bool): Flag to indicate that the SDK is running in CLI-only mode.
- x_disable_machine_info (bool): Flag to disable automatic machine info collection.
- x_disable_meta (bool): Flag to disable the collection of system metadata.
- x_disable_setproctitle (bool): Flag to disable using setproctitle for the internal process in the legacy service.
    This is deprecated and will be removed in future versions.
- x_disable_stats (bool): Flag to disable the collection of system metrics.
- x_disable_viewer (bool): Flag to disable the early viewer query.
- x_executable (Optional): Path to the Python executable.
- x_extra_http_headers (Optional): Additional headers to add to all outgoing HTTP requests.
- x_file_stream_max_bytes (Optional): An approximate maximum request size for the filestream API.
    Its purpose is to prevent HTTP requests from failing due to
    containing too much data. This number is approximate:
    requests will be slightly larger.
- x_file_stream_max_line_bytes (Optional): Maximum line length for filestream JSONL files.
- x_file_stream_retry_max (Optional): Max number of retries for filestream operations.
- x_file_stream_retry_wait_max_seconds (Optional): Maximum wait time between retries for filestream operations.
- x_file_stream_retry_wait_min_seconds (Optional): Minimum wait time between retries for filestream operations.
- x_file_stream_timeout_seconds (Optional): Timeout in seconds for individual filestream HTTP requests.
- x_file_stream_transmit_interval (Optional): Interval in seconds between filestream transmissions.
- x_file_transfer_retry_max (Optional): Max number of retries for file transfer operations.
- x_file_transfer_retry_wait_max_seconds (Optional): Maximum wait time between retries for file transfer operations.
- x_file_transfer_retry_wait_min_seconds (Optional): Minimum wait time between retries for file transfer operations.
- x_file_transfer_timeout_seconds (Optional): Timeout in seconds for individual file transfer HTTP requests.
- x_files_dir (Optional): Override setting for the computed files_dir..
- x_flow_control_custom (Optional): Flag indicating custom flow control for filestream.
    TODO: Not implemented in wandb-core.
- x_flow_control_disabled (Optional): Flag indicating flow control is disabled for filestream.
    TODO: Not implemented in wandb-core.
- x_graphql_retry_max (Optional): Max number of retries for GraphQL operations.
- x_graphql_retry_wait_max_seconds (Optional): Maximum wait time between retries for GraphQL operations.
- x_graphql_retry_wait_min_seconds (Optional): Minimum wait time between retries for GraphQL operations.
- x_graphql_timeout_seconds (Optional): Timeout in seconds for individual GraphQL requests.
- x_internal_check_process (float): Interval for internal process health checks in seconds.
- x_jupyter_name (Optional): Name of the Jupyter notebook.
- x_jupyter_path (Optional): Path to the Jupyter notebook.
- x_jupyter_root (Optional): Root directory of the Jupyter notebook.
- x_label (Optional): Label to assign to system metrics and console logs collected for the run.
    This is used to group data by on the frontend and can be used to distinguish data
    from different processes in a distributed training job.
- x_live_policy_rate_limit (Optional): Rate limit for live policy updates in seconds.
- x_live_policy_wait_time (Optional): Wait time between live policy updates in seconds.
- x_log_level (int): Logging level for internal operations.
- x_network_buffer (Optional): Size of the network buffer used in flow control.
    TODO: Not implemented in wandb-core.
- x_primary (bool): Determines whether to save internal wandb files and metadata.
    In a distributed setting, this is useful for avoiding file overwrites
    from secondary processes when only system metrics and logs are needed,
    as the primary process handles the main logging.
- x_proxies (Optional): Custom proxy servers for requests to W&B.
    This is deprecated and will be removed in future versions.
    Please use `http_proxy` and `https_proxy` instead.
- x_require_legacy_service (bool): Force the use of legacy wandb service.
- x_runqueue_item_id (Optional): ID of the Launch run queue item being processed.
- x_save_requirements (bool): Flag to save the requirements file.
- x_server_side_derived_summary (bool): Flag to delegate automatic computation of summary from history to the server.
    This does not disable user-provided summary updates.
- x_server_side_expand_glob_metrics (bool): Flag to delegate glob matching of metrics in define_metric to the server.
    If the server does not support this, the client will perform the glob matching.
- x_service_transport (Optional): Transport method for communication with the wandb service.
- x_service_wait (float): Time in seconds to wait for the wandb-core internal service to start.
- x_skip_transaction_log (bool): Whether to skip saving the run events to the transaction log.
    This is only relevant for online runs. Can be used to reduce the amount of
    data written to disk.
    Should be used with caution, as it removes the gurantees about
    recoverability.
- x_start_time (Optional): The start time of the run in seconds since the Unix epoch.
- x_stats_buffer_size (int): Number of system metric samples to buffer in memory in the wandb-core process.
    Can be accessed via run._system_metrics.
- x_stats_coreweave_metadata_base_url (str): The scheme and hostname for contacting the CoreWeave metadata server.
    Only accessible from within a CoreWeave cluster.
- x_stats_coreweave_metadata_endpoint (str): The relative path on the CoreWeave metadata server to which to make requests.
    This must not include the schema and hostname prefix.
    Only accessible from within a CoreWeave cluster.
- x_stats_dcgm_exporter (Optional): Endpoint to extract Nvidia DCGM metrics from.
    Two options are supported:
    - Extract DCGM-related metrics from a query to the Prometheus `/api/v1/query` endpoint.
    It is a common practice to aggregate metrics reported by the instances of the DCGM Exporter
    running on different nodes in a cluster using Prometheus.
    - TODO: Parse metrics directly from the `/metrics` endpoint of the DCGM Exporter.
    Examples:
    - `http://localhost:9400/api/v1/query?query=DCGM_FI_DEV_GPU_TEMP{node="l1337", cluster="globular"}`.
    - TODO: `http://192.168.0.1:9400/metrics`.
- x_stats_disk_paths (Optional): System paths to monitor for disk usage.
- x_stats_gpu_device_ids (Optional): GPU device indices to monitor.
    If not set, captures metrics for all GPUs.
    Assumes 0-based indexing matching CUDA/ROCm device enumeration.
- x_stats_neuron_monitor_config_path (Optional): Path to the default config file for the neuron-monitor tool.
    This is used to monitor AWS Trainium devices.
- x_stats_open_metrics_endpoints (Optional): OpenMetrics `/metrics` endpoints to monitor for system metrics.
- x_stats_open_metrics_filters (Union): Filter to apply to metrics collected from OpenMetrics `/metrics` endpoints.
    Supports two formats:
    - {"metric regex pattern, including endpoint name as prefix": {"label": "label value regex pattern"}}
    - ("metric regex pattern 1", "metric regex pattern 2", ...)
- x_stats_open_metrics_http_headers (Optional): HTTP headers to add to OpenMetrics requests.
- x_stats_pid (int): PID of the process that started the wandb-core process to collect system stats for.
- x_stats_sampling_interval (float): Sampling interval for the system monitor in seconds.
- x_sync (bool): Flag to indicate whether we are syncing a run from the transaction log.
- x_update_finish_state (bool): Flag to indicate whether this process can update the run's final state on the server.
    Set to False in distributed training when only the main process should determine the final state.