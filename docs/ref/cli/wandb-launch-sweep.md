# wandb launch-sweep

**Usage**

`wandb launch-sweep [OPTIONS] [CONFIG]`

**Summary**

Run a W&B launch sweep (Experimental).

**Options**

| **Option** | **Description** |
| :--- | :--- |
| -q, --queue | The name of a queue to push the sweep to |
| -p, --project | Name of the project which the agent will watch.   If passed in, will override the project value |
| -e, --entity | The entity to use. Defaults to current logged-in   user |
| -r, --resume_id | Resume a launch sweep by passing an 8-char sweep   id. Queue required |
| -n, --num_workers | Number of concurrent jobs a scheduler can run |
| --help | Show this message and exit. |

