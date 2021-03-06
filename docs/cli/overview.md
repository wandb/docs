---
title: Command Line Overview
sidebar_label: Command Line Overview
---

After running `pip install wandb` you should have a new command available, **wandb**. The following sub-commands are available:

| Sub-command | Description                                                       |
| ----------- | ----------------------------------------------------------------- |
| docs        | Open documentation in a browser                                   |
| init        | Configure a directory with W&B                                    |
| login       | Login to W&B                                                      |
| off         | Disable W&B in this directory, useful for testing                 |
| on          | Ensure W&B is enabled in this directory                           |
| docker      | Run a docker image, mount cwd, and ensure wandb is installed      |
| docker-run  | Add W&B environment variables to a docker run command             |
| projects    | List projects                                                     |
| pull        | Pull files for a run from W&B                                     |
| restore     | Restore code and config state for a run                           |
| run         | Launch a job, required on Windows                                 |
| runs        | List runs in a project                                            |
| sync        | Sync a local directory containing tfevents or previous runs files |
| status      | List current directory status                                     |
| sweep       | Create a new sweep given a YAML definition                        |
| agent       | Start an agent to run programs in the sweep                       |
