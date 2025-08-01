---
description: Set W&B environment variables.
menu:
  default:
    identifier: environment-variables
    parent: experiments
title: Environment variables
weight: 9
---

When you're running a script in an automated environment, you can control W&B with environment variables set before the script runs or within the script.

```bash
# This is secret and shouldn't be checked into version control
WANDB_API_KEY=$YOUR_API_KEY
# Name and notes optional
WANDB_NAME="My first run"
WANDB_NOTES="Smaller learning rate, more regularization."
```

```bash
# Only needed if you don't check in the wandb/settings file
WANDB_ENTITY=$username
WANDB_PROJECT=$project
```

```python
# If you don't want your script to sync to the cloud
os.environ["WANDB_MODE"] = "offline"

# Add sweep ID tracking to Run objects and related classes
os.environ["WANDB_SWEEP_ID"] = "b05fq58z"
```

## Optional environment variables

Use these optional environment variables to do things like set up authentication on remote machines.

| Variable name | Usage |
| --------------------------- | ---------- |
| `WANDB_ANONYMOUS` | Set this to `allow`, `never`, or `must` to let users create anonymous runs with secret urls. |
| `WANDB_API_KEY` | Sets the authentication key associated with your account. You can find your key on [your settings page](https://app.wandb.ai/settings).  This must be set if `wandb login` hasn't been run on the remote machine.               |
| `WANDB_BASE_URL` | If you're using [wandb/local]({{< relref "/guides/hosting/" >}}) you should set this environment variable to `http://YOUR_IP:YOUR_PORT` |
| `WANDB_CACHE_DIR` | This defaults to \~/.cache/wandb, you can override this location with this environment variable |
| `WANDB_CONFIG_DIR` | This defaults to \~/.config/wandb, you can override this location with this environment variable |
| `WANDB_CONFIG_PATHS` | Comma separated list of yaml files to load into wandb.config. See [config]({{< relref "./config.md#file-based-configs" >}}). |
| `WANDB_CONSOLE` | Set this to "off" to disable stdout / stderr logging. This defaults to "on" in environments that support it. |
| `WANDB_DATA_DIR` | Where to upload staging artifacts. The default location depends on your platform, because it uses the value of `user_data_dir` from the `platformdirs` Python package. Make sure this directory exists and the running user has permission to write to it. |
| `WANDB_DIR` | Where to store all generated files. If unset, defaults to the `wandb` directory relative to your training script. Make sure this directory exists and the running user has permission to write to it. This does not control the location of downloaded artifacts, which you can set using the `WANDB_ARTIFACT_DIR` environment variable. |
| `WANDB_ARTIFACT_DIR` | Where to store all downloaded artifacts. If unset, defaults to the `artifacts` directory relative to your training script. Make sure this directory exists and the running user has permission to write to it. This does not control the location of generated metadata files, which you can set using the `WANDB_DIR` environment variable. |
| `WANDB_DISABLE_GIT` | Prevent wandb from probing for a git repository and capturing the latest commit / diff. |
| `WANDB_DISABLE_CODE` | Set this to true to prevent wandb from saving notebooks or git diffs. We'll still save the current commit if we're in a git repo. |
| `WANDB_DOCKER` | Set this to a docker image digest to enable restoring of runs. This is set automatically with the wandb docker command. You can obtain an image digest by running `wandb docker my/image/name:tag --digest` |
| `WANDB_ENTITY` | The entity associated with your run. If you have run `wandb init` in the directory of your training script, it will create a directory named _wandb_ and will save a default entity which can be checked into source control. If you don't want to create that file or want to override the file you can use the environmental variable. |
| `WANDB_ERROR_REPORTING` | Set this to false to prevent wandb from logging fatal errors to its error tracking system. |
| `WANDB_HOST` | Set this to the hostname you want to see in the wandb interface if you don't want to use the system provided hostname |
| `WANDB_IGNORE_GLOBS` | Set this to a comma separated list of file globs to ignore. These files will not be synced to the cloud. |
| `WANDB_JOB_NAME` | Specify a name for any jobs created by `wandb`. |
| `WANDB_JOB_TYPE` | Specify the job type, like "training" or "evaluation" to indicate different types of runs. See [grouping]({{< relref "/guides/models/track/runs/grouping.md" >}}) for more info. |
| `WANDB_MODE` | If you set this to "offline" wandb will save your run metadata locally and not sync to the server. If you set this to `disabled` wandb will turn off completely. |
| `WANDB_NAME` | The human-readable name of your run. If not set it will be randomly generated for you |
| `WANDB_NOTEBOOK_NAME` | If you're running in jupyter you can set the name of the notebook with this variable. We attempt to auto detect this. |
| `WANDB_NOTES` | Longer notes about your run. Markdown is allowed and you can edit this later in the UI. |
| `WANDB_PROJECT` | The project associated with your run. This can also be set with `wandb init`, but the environmental variable will override the value. |
| `WANDB_RESUME` | By default this is set to _never_. If set to _auto_ wandb will automatically resume failed runs. If set to _must_ forces the run to exist on startup. If you want to always generate your own unique ids, set this to _allow_ and always set `WANDB_RUN_ID`. |
| `WANDB_RUN_GROUP` | Specify the experiment name to automatically group runs together. See [grouping]({{< relref "/guides/models/track/runs/grouping.md" >}}) for more info. |
| `WANDB_RUN_ID` | Set this to a globally unique string (per project) corresponding to a single run of your script. It must be no longer than 64 characters. All non-word characters will be converted to dashes. This can be used to resume an existing run in cases of failure. |
| `WANDB_QUIET` | Set this to `true` to limit statements logged to standard output to critical statements only. If this is set all logs will be written to `$WANDB_DIR/debug.log`. |
| `WANDB_SILENT` | Set this to `true` to silence wandb log statements. This is useful for scripted commands. If this is set all logs will be written to `$WANDB_DIR/debug.log`. |
| `WANDB_SHOW_RUN` | Set this to `true` to automatically open a browser with the run url if your operating system supports it. |
| `WANDB_SWEEP_ID` | Add sweep ID tracking to `Run` objects and related classes, and display in the UI. |
| `WANDB_TAGS` | A comma separated list of tags to be applied to the run. |
| `WANDB_USERNAME` | The username of a member of your team associated with the run. This can be used along with a service account API key to enable attribution of automated runs to members of your team. |
| `WANDB_USER_EMAIL` | The email of a member of your team associated with the run. This can be used along with a service account API key to enable attribution of automated runs to members of your team. |

## Singularity environments

If you're running containers in [Singularity](https://singularity.lbl.gov/index.html) you can pass environment variables by pre-pending the above variables with `SINGULARITYENV_`. More details about Singularity environment variables can be found [here](https://singularity.lbl.gov/docs-environment-metadata#environment).

## Running on AWS

If you're running batch jobs in AWS, it's easy to authenticate your machines with your W&B credentials. Get your API key from your [settings page](https://app.wandb.ai/settings), and set the `WANDB_API_KEY` environment variable in the [AWS batch job spec](https://docs.aws.amazon.com/batch/latest/userguide/job_definition_parameters.html#parameters).
