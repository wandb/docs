# Command Line Interface

**Usage**

`wandb [OPTIONS] COMMAND [ARGS]...`


**Options**

| **Option** | **Description** |
| :--- | :--- |
| --version | Show the version and exit. |
| --help | Show this message and exit. |

**Commands**

| **Command** | **Description** |
| :--- | :--- |
| agent | Run the W&B agent |
| artifact | Commands for interacting with artifacts |
| controller | Run the W&B local sweep controller |
| disabled | Disable W&B. |
| docker | Run your code in a docker container. |
| docker-run | Wrap `docker run` and adds WANDB_API_KEY and WANDB_DOCKER... |
| enabled | Enable W&B. |
| import | Commands for importing data from other systems |
| init | Configure a directory with Weights & Biases |
| launch | Launch or queue a W&B Job. |
| launch-agent | Run a W&B launch agent. |
| login | Login to Weights & Biases |
| offline | Disable W&B sync |
| online | Enable W&B sync |
| pull | Pull files from Weights & Biases |
| restore | Restore code, config and docker state for a run |
| scheduler | Run a W&B launch sweep scheduler (Experimental) |
| server | Commands for operating a local W&B server |
| status | Show configuration settings |
| sweep | Create a sweep |
| sync | Upload an offline training directory to W&B |
| verify | Verify your local instance |
