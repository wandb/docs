---
title: wandb job create
---

Create a job from a source, without a wandb run.

Jobs can be of three types, git, code, or image.

git: A git source, with an entrypoint either in the path or provided explicitly pointing to the main python executable. code: A code path, containing a requirements.txt file. image: A docker image.

## Usage

```bash
wandb create JOB_TYPE PATH [OPTIONS]
```

## Arguments

| Argument | Description | Required |
| :--- | :--- | :--- |
| `JOB_TYPE` | No description available | Yes |
| `PATH` | No description available | Yes |

## Options

| Option | Description |
| :--- | :--- |
| `--project`, `-p` | The project you want to list jobs from. |
| `--entity`, `-e` | The entity the jobs belong to |
| `--name`, `-n` | Name for the job |
| `--description`, `-d` | Description for the job |
| `--alias`, `-a` | Alias for the job |
| `--entry-point`, `-E` | Entrypoint to the script, including an executable and an entrypoint file. Required for code or repo jobs. If --build-context is provided, paths in the entrypoint command will be relative to the build context. |
| `--git-hash`, `-g` | Commit reference to use as the source for git jobs |
| `--runtime`, `-r` | Python runtime to execute the job |
| `--build-context`, `-b` | Path to the build context from the root of the job source code. If provided, this is used as the base path for the Dockerfile and entrypoint. |
| `--base-image`, `-B` | Base image to use for the job. Incompatible with image jobs. |
| `--dockerfile`, `-D` | Path to the Dockerfile for the job. If --build-context is provided, the Dockerfile path will be relative to the build context. |
| `--service`, `-s` | Service configurations in format serviceName=policy. Valid policies: always, never |
| `--schema` | Path to the schema file for the job. |
