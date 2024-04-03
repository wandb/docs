# wandb job create

**Usage**

`wandb job create [OPTIONS] {git|code|image} PATH`

**Summary**

Create a job from a source, without a wandb run.

Jobs can be of three types, git, code, or image.

git: A git source, with an entrypoint either in the path or provided
explicitly pointing to the main python executable. code: A code path,
containing a requirements.txt file. image: A docker image.

**Options**

| **Option** | **Description** |
| :--- | :--- |
| -p, --project | The project you want to list jobs from. |
| -e, --entity | The entity the jobs belong to |
| -n, --name | Name for the job |
| -d, --description | Description for the job |
| -a, --alias | Alias for the job |
| --entry-point | Entrypoint to the script, including an executable   and an entrypoint file. Required for code or repo jobs |
| -g, --git-hash | Hash to a specific git commit. |
| -r, --runtime | Python runtime to execute the job |

