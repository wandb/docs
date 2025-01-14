---
title: Import & Export API
---
<!-- Insert buttons and diff -->

{{< cta-button githubLink="https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/__init__.py" >}}

## Classes

[`class Api`](./api/): Used for querying the wandb server.

[`class File`](./file/): File is a class associated with a file saved by wandb.

[`class Files`](./files/): An iterable collection of `File` objects.

[`class Job`](./job/)

[`class Project`](./project/): A project is a namespace for runs.

[`class Projects`](./projects/): An iterable collection of `Project` objects.

[`class QueuedRun`](./queuedrun/): A single queued run associated with an entity and project. Call `run = queued_run.wait_until_running()` or `run = queued_run.wait_until_finished()` to access the run.

[`class Run`](./run/): A single run associated with an entity and project.

[`class RunQueue`](./runqueue/)

[`class Runs`](./runs/): An iterable collection of runs associated with a project and optional filter.

[`class Sweep`](./sweep/): A set of runs associated with a sweep.
