# wandb launch-agent

**Usage**

`wandb launch-agent [OPTIONS]`

**Summary**

Run a W&B launch agent.

**Options**

| **Option** | **Description** |
| :--- | :--- |
| -q, --queue <queue(s)> | The name of a queue for the agent to watch. Multiple   -q flags supported. |
| -p, --project | Name of the project which the agent will watch. If   passed in, will override the project value passed in |
| -e, --entity | The entity to use. Defaults to current logged-in   user |
| -j, --max-jobs | The maximum number of launch jobs this agent can run   in parallel. Defaults to 1. Set to -1 for no upper |
| -c, --config | path to the agent config yaml to use |
| --help | Show this message and exit. |

