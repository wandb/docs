# wandb launch

**Usage**

`wandb launch [OPTIONS]`

**Summary**

Launch or queue a W&B Job. See https://wandb.me/launch

**Options**

> NOTE: Any values passed in via command line arguments take precedence over values from config files.

| **Option**                       | **Description**                                                                                                                                                                                                                                                                    |
| :------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-j, --job (str)`                | Name of the job to launch. If passed in, `wandb launch` does not require a URI.                                                                                                                                                                                                    |
| `--entry-point (str)`            | Entry point within project. Default is `main`. If the entry point is not found, attempt to run the project file with the specified name as a script, using `python` to run `.py` files and the default shell (specified by environment variable `$SHELL`) to run `.sh` files.      |
| `-a, --args-list (name)=(value)` | An argument for the run, of the form `-a name=value`. May be passed multiple times. Provided arguments that are not in the list of arguments for an entrypoint will be passed to the corresponding entrypoint as command-line arguments in the form `--name value`.                |
| `--name (str)`                   | Name of the run under which to launch the run. Defaults to a random name.                                                                                                                                                                                                          |
| `-e, --entity (str)`             | Name of the target entity which the new run will be sent to. Defaults to the entity set inside `wandb/settings` folder.                                                                                                                                                            |
| `-p, --project (str)`            | Name of the target project which the new run will be sent to. Defaults to the project name given by the source URI, or the git repo name for github URIs.                                                                                                                          |
| `-r, --resource (str)`           | Execution resource to use for run. Supported values: `local-process`, `local-container`, `kubernetes`, `sagemaker`, `gcp-vertex`. This is now a required parameter if pushing to a queue with no resource configuration. If passed in, will override the queue's default resource. |
| `-d, --docker-image (str)`       | Docker image to use, in the form `name:tag`.                                                                                                                                                                                                                                       |
| `-c, --config (str)`             | Path to JSON file (must end in `.json`) or JSON string which will be passed as a launch config.                                                                                                                                                                                    |
| `-q, --queue [str]`              | Push this run to a queue, optionally passing a queue name.  If unset, launches a single run directly. If set without passing a queue name (`--queue`), defaults to `default`.                                                                                                      |
| `--async`                        | Invoke this run asynchronously. Defaults to `false`, i.e. unless `--async` is set, `wandb launch` will wait for the job to finish. This option is incompatible with `--queue`; asynchronous options when running with an agent should be set on `wandb launch-agent`.              |
| `--resource-args`                | Path to JSON file (must end in `.json`) or JSON string which will be passed as resource args to the compute resource. The exact content which should be provided is different for each execution backend. See documentation for layout of this file.                               |
| `-pq`, `--project-queue`         | Name of the project containing the queue to push to. If unset, defaults to entity level.                                                                                                                                                                                           |
| `--help`                         | Show this message and exit.                                                                                                                                                                                                                                                        |
