# Import & Export API

<!-- Insert buttons and diff -->


<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.15.7/wandb/__init__.py' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


Use the Public API to export or update data that you have saved to W&B.

Before using this API, you'll want to log data from your script — check the
[Quickstart](https://docs.wandb.ai/quickstart) for more details.

You might use the Public API to

- update metadata or metrics for an experiment after it has been completed,
- pull down your results as a dataframe for post-hoc analysis in a Jupyter notebook, or
- check your saved model artifacts for those tagged as `ready-to-deploy`.

For more on using the Public API, check out [our guide](https://docs.wandb.com/guides/track/public-api-guide).

## Classes

[`class Api`](./api.md): Used for querying the wandb server.

[`class File`](./file.md): File is a class associated with a file saved by wandb.

[`class Files`](./files.md): An iterable collection of `File` objects.

[`class Project`](./project.md): A project is a namespace for runs.

[`class Projects`](./projects.md): An iterable collection of `Project` objects.

[`class Run`](./run.md): A single run associated with an entity and project.

[`class Runs`](./runs.md): An iterable collection of runs associated with a project and optional filter.

[`class Sweep`](./sweep.md): A set of runs associated with a sweep.
