# wandb launch

**Usage**

`wandb launch [OPTIONS]`

**Summary**

Launch or queue a W&B Job. See https://wandb.me/launch

**Options**

| **Option** | **Description** |
| :--- | :--- |
| -j, --job (str) | Name of the job to launch. If passed in,   launch does not require a uri. |
| --entry-point | Entry point within project. [default: main].   If the entry point is not found, attempts torun the project file with the specified name   as a script, using 'python' to run .py filesand the default shell (specified by   environment variable $SHELL) to run .shfiles. If passed in, will override the   entrypoint value passed in using a configfile. |
| --name | Name of the run under which to launch the   run. If not specified, a random run namewill be used to launch run. If passed in,   will override the name passed in using aconfig file. |
| -e, --entity (str) | Name of the target entity which the new run   will be sent to. Defaults to using theentity set by local wandb/settings folder.   If passed in, will override the entity valuepassed in using a config file. |
| -p, --project (str) | Name of the target project which the new run   will be sent to. Defaults to using theproject name given by the source uri or for   github runs, the git repo name. If passedin, will override the project value passed   in using a config file. |
| -r, --resource | Execution resource to use for run. Supported   values: 'local-process', 'local-container','kubernetes', 'sagemaker', 'gcp-vertex'.   This is now a required parameter if pushingto a queue with no resource configuration.   If passed in, will override the resourcevalue passed in using a config file. |
| -d, --docker-image | Specific docker image you'd like to use. Inthe form name:tag. If passed in, will   override the docker image value passed inusing a config file. |
| -c, --config | Path to JSON file (must end in '.json') or   JSON string which will be passed as a launchconfig. Dictation how the launched run will   be configured. |
| -q, --queue | Name of run queue to push to. If none,   launches single run directly. If suppliedwithout an argument (`--queue`), defaults to   queue 'default'. Else, if name supplied,specified run queue must exist under the   project and entity supplied. |
| --async | Flag to run the job asynchronously. Defaults   to false, i.e. unless --async is set, wandblaunch will wait for the job to finish. This   option is incompatible with --queue;asynchronous options when running with an   agent should be set on wandb launch-agent. |
| --resource-args | Path to JSON file (must end in '.json') or   JSON string which will be passed as resourceargs to the compute resource. The exact   content which should be provided isdifferent for each execution backend. See   documentation for layout of this file. |

