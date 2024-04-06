# Api

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.16.6/wandb/apis/public/api.py#L98-L1044' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


Used for querying the wandb server.

```python
Api(
    overrides=None,
    timeout: Optional[int] = None,
    api_key: Optional[str] = None
) -> None
```

#### Examples:

Most common way to initialize

```
>>> wandb.Api()
```

| Arguments |  |
| :--- | :--- |
|  `overrides` |  (dict) You can set `base_url` if you are using a wandb server other than https://api.wandb.ai. You can also set defaults for `entity`, `project`, and `run`. |

| Attributes |  |
| :--- | :--- |

## Methods

### `artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.6/wandb/apis/public/api.py#L941-L965)

```python
artifact(
    name, type=None
)
```

Return a single artifact by parsing path in the form `entity/project/name`.

| Arguments |  |
| :--- | :--- |
|  `name` |  (str) An artifact name. May be prefixed with entity/project. Valid names can be in the following forms: name:version name:alias |
|  `type` |  (str, optional) The type of artifact to fetch. |

| Returns |  |
| :--- | :--- |
|  A `Artifact` object. |

### `artifact_collection`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.6/wandb/apis/public/api.py#L910-L924)

```python
artifact_collection(
    type_name: str,
    name: str
)
```

Return a single artifact collection by type and parsing path in the form `entity/project/name`.

| Arguments |  |
| :--- | :--- |
|  `name` |  (str) An artifact collection name. May be prefixed with entity/project. |
|  `type` |  (str) The type of artifact collection to fetch. |

| Returns |  |
| :--- | :--- |
|  An `ArtifactCollection` object. |

### `artifact_collections`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.6/wandb/apis/public/api.py#L903-L908)

```python
artifact_collections(
    project_name: str,
    type_name: str,
    per_page=50
)
```

### `artifact_type`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.6/wandb/apis/public/api.py#L898-L901)

```python
artifact_type(
    type_name, project=None
)
```

### `artifact_types`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.6/wandb/apis/public/api.py#L893-L896)

```python
artifact_types(
    project=None
)
```

### `artifact_versions`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.6/wandb/apis/public/api.py#L926-L932)

```python
artifact_versions(
    type_name, name, per_page=50
)
```

Deprecated, use artifacts(type_name, name) instead.

### `artifacts`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.6/wandb/apis/public/api.py#L934-L939)

```python
artifacts(
    type_name, name, per_page=50
)
```

### `create_project`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.6/wandb/apis/public/api.py#L273-L274)

```python
create_project(
    name: str,
    entity: str
)
```

### `create_report`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.6/wandb/apis/public/api.py#L282-L297)

```python
create_report(
    project: str,
    entity: str = "",
    title: Optional[str] = "Untitled Report",
    description: Optional[str] = "",
    width: Optional[str] = "readable",
    blocks: Optional['wandb.apis.reports.util.Block'] = None
) -> "wandb.apis.reports.Report"
```

### `create_run`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.6/wandb/apis/public/api.py#L276-L280)

```python
create_run(
    **kwargs
)
```

Create a new run.

### `create_run_queue`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.6/wandb/apis/public/api.py#L299-L409)

```python
create_run_queue(
    name: str,
    type: "public.RunQueueResourceType",
    entity: Optional[str] = None,
    prioritization_mode: Optional['public.RunQueuePrioritizationMode'] = None,
    config: Optional[dict] = None,
    template_variables: Optional[dict] = None
) -> "public.RunQueue"
```

Create a new run queue (launch).

| Arguments |  |
| :--- | :--- |
|  `name` |  (str) Name of the queue to create |
|  `type` |  (str) Type of resource to be used for the queue. One of "local-container", "local-process", "kubernetes", "sagemaker", or "gcp-vertex". |
|  `entity` |  (str) Optional name of the entity to create the queue. If None, will use the configured or default entity. |
|  `prioritization_mode` |  (str) Optional version of prioritization to use. Either "V0" or None |
|  `config` |  (dict) Optional default resource configuration to be used for the queue. Use handlebars (eg. `{{var}}`) to specify template variables. |
|  `template_variables` |  (dict) A dictionary of template variable schemas to be used with the config. Expected format of: `{ "var-name": { "schema": { "type": ("string", "number", or "integer"), "default": (optional value)`, "minimum": (optional minimum), "maximum": (optional maximum), "enum": [..."(options)"] } } } |

| Returns |  |
| :--- | :--- |
|  The newly created `RunQueue` |

| Raises |  |
| :--- | :--- |
|  ValueError if any of the parameters are invalid wandb.Error on wandb API errors |

### `create_team`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.6/wandb/apis/public/api.py#L690-L700)

```python
create_team(
    team, admin_username=None
)
```

Create a new team.

| Arguments |  |
| :--- | :--- |
|  `team` |  (str) The name of the team |
|  `admin_username` |  (str) optional username of the admin user of the team, defaults to the current user. |

| Returns |  |
| :--- | :--- |
|  A `Team` object |

### `create_user`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.6/wandb/apis/public/api.py#L427-L437)

```python
create_user(
    email, admin=(False)
)
```

Create a new user.

| Arguments |  |
| :--- | :--- |
|  `email` |  (str) The email address of the user |
|  `admin` |  (bool) Whether this user should be a global instance admin |

| Returns |  |
| :--- | :--- |
|  A `User` object |

### `flush`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.6/wandb/apis/public/api.py#L504-L511)

```python
flush()
```

Flush the local cache.

The api object keeps a local cache of runs, so if the state of the run may
change while executing your script you must clear the local cache with
`api.flush()` to get the latest values associated with the run.

### `from_path`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.6/wandb/apis/public/api.py#L513-L567)

```python
from_path(
    path
)
```

Return a run, sweep, project or report from a path.

#### Examples:

```
project = api.from_path("my_project")
team_project = api.from_path("my_team/my_project")
run = api.from_path("my_team/my_project/runs/id")
sweep = api.from_path("my_team/my_project/sweeps/id")
report = api.from_path("my_team/my_project/reports/My-Report-Vm11dsdf")
```

| Arguments |  |
| :--- | :--- |
|  `path` |  (str) The path to the project, run, sweep or report |

| Returns |  |
| :--- | :--- |
|  A `Project`, `Run`, `Sweep`, or `BetaReport` instance. |

| Raises |  |
| :--- | :--- |
|  wandb.Error if path is invalid or the object doesn't exist |

### `job`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.6/wandb/apis/public/api.py#L967-L975)

```python
job(
    name, path=None
)
```

### `list_jobs`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.6/wandb/apis/public/api.py#L977-L1044)

```python
list_jobs(
    entity, project
)
```

### `load_report`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.6/wandb/apis/public/api.py#L411-L425)

```python
load_report(
    path: str
) -> "wandb.apis.reports.Report"
```

Get report at a given path.

| Arguments |  |
| :--- | :--- |
|  `path` |  (str) Path to the target report in the form `entity/project/reports/reportId`. You can get this by copy-pasting the URL after your wandb url. For example: `megatruong/report-editing/reports/My-fabulous-report-title--VmlldzoxOTc1Njk0` |

| Returns |  |
| :--- | :--- |
|  A `BetaReport` object which represents the report at `path` |

| Raises |  |
| :--- | :--- |
|  wandb.Error if path is invalid |

### `project`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.6/wandb/apis/public/api.py#L654-L657)

```python
project(
    name, entity=None
)
```

### `projects`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.6/wandb/apis/public/api.py#L629-L652)

```python
projects(
    entity=None, per_page=200
)
```

Get projects for a given entity.

| Arguments |  |
| :--- | :--- |
|  `entity` |  (str) Name of the entity requested. If None, will fall back to default entity passed to `Api`. If no default entity, will raise a `ValueError`. |
|  `per_page` |  (int) Sets the page size for query pagination. None will use the default size. Usually there is no reason to change this. |

| Returns |  |
| :--- | :--- |
|  A `Projects` object which is an iterable collection of `Project` objects. |

### `queued_run`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.6/wandb/apis/public/api.py#L838-L859)

```python
queued_run(
    entity, project, queue_name, run_queue_item_id, project_queue=None,
    priority=None
)
```

Return a single queued run based on the path.

Parses paths of the form entity/project/queue_id/run_queue_item_id.

### `reports`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.6/wandb/apis/public/api.py#L659-L688)

```python
reports(
    path="", name=None, per_page=50
)
```

Get reports for a given project path.

WARNING: This api is in beta and will likely change in a future release

| Arguments |  |
| :--- | :--- |
|  `path` |  (str) path to project the report resides in, should be in the form: "entity/project" |
|  `name` |  (str) optional name of the report requested. |
|  `per_page` |  (int) Sets the page size for query pagination. None will use the default size. Usually there is no reason to change this. |

| Returns |  |
| :--- | :--- |
|  A `Reports` object which is an iterable collection of `BetaReport` objects. |

### `run`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.6/wandb/apis/public/api.py#L821-L836)

```python
run(
    path=""
)
```

Return a single run by parsing path in the form entity/project/run_id.

| Arguments |  |
| :--- | :--- |
|  `path` |  (str) path to run in the form `entity/project/run_id`. If `api.entity` is set, this can be in the form `project/run_id` and if `api.project` is set this can just be the run_id. |

| Returns |  |
| :--- | :--- |
|  A `Run` object. |

### `run_queue`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.6/wandb/apis/public/api.py#L861-L874)

```python
run_queue(
    entity, name
)
```

Return the named `RunQueue` for entity.

To create a new `RunQueue`, use `wandb.Api().create_run_queue(...)`.

### `runs`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.6/wandb/apis/public/api.py#L743-L819)

```python
runs(
    path: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    order: str = "-created_at",
    per_page: int = 50,
    include_sweeps: bool = (True)
)
```

Return a set of runs from a project that match the filters provided.

You can filter by `config.*`, `summary_metrics.*`, `tags`, `state`, `entity`, `createdAt`, etc.

#### Examples:

Find runs in my_project where config.experiment_name has been set to "foo"

```
api.runs(path="my_entity/my_project", filters={"config.experiment_name": "foo"})
```

Find runs in my_project where config.experiment_name has been set to "foo" or "bar"

```
api.runs(
    path="my_entity/my_project",
    filters={"$or": [{"config.experiment_name": "foo"}, {"config.experiment_name": "bar"}]}
)
```

Find runs in my_project where config.experiment_name matches a regex (anchors are not supported)

```
api.runs(
    path="my_entity/my_project",
    filters={"config.experiment_name": {"$regex": "b.*"}}
)
```

Find runs in my_project where the run name matches a regex (anchors are not supported)

```
api.runs(
    path="my_entity/my_project",
    filters={"display_name": {"$regex": "^foo.*"}}
)
```

Find runs in my_project sorted by ascending loss

```
api.runs(path="my_entity/my_project", order="+summary_metrics.loss")
```

| Arguments |  |
| :--- | :--- |
|  `path` |  (str) path to project, should be in the form: "entity/project" |
|  `filters` |  (dict) queries for specific runs using the MongoDB query language. You can filter by run properties such as config.key, summary_metrics.key, state, entity, createdAt, etc. For example: `{"config.experiment_name": "foo"}` would find runs with a config entry of experiment name set to "foo" You can compose operations to make more complicated queries, see Reference for the language is at https://docs.mongodb.com/manual/reference/operator/query |
|  `order` |  (str) Order can be `created_at`, `heartbeat_at`, `config.*.value`, or `summary_metrics.*`. If you prepend order with a + order is ascending. If you prepend order with a - order is descending (default). The default order is run.created_at from newest to oldest. |

| Returns |  |
| :--- | :--- |
|  A `Runs` object, which is an iterable collection of `Run` objects. |

### `sweep`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.6/wandb/apis/public/api.py#L876-L891)

```python
sweep(
    path=""
)
```

Return a sweep by parsing path in the form `entity/project/sweep_id`.

| Arguments |  |
| :--- | :--- |
|  `path` |  (str, optional) path to sweep in the form entity/project/sweep_id. If `api.entity` is set, this can be in the form project/sweep_id and if `api.project` is set this can just be the sweep_id. |

| Returns |  |
| :--- | :--- |
|  A `Sweep` object. |

### `sync_tensorboard`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.6/wandb/apis/public/api.py#L439-L461)

```python
sync_tensorboard(
    root_dir, run_id=None, project=None, entity=None
)
```

Sync a local directory containing tfevent files to wandb.

### `team`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.6/wandb/apis/public/api.py#L702-L703)

```python
team(
    team
)
```

### `user`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.6/wandb/apis/public/api.py#L705-L725)

```python
user(
    username_or_email
)
```

Return a user from a username or email address.

Note: This function only works for Local Admins, if you are trying to get your own user object, please use `api.viewer`.

| Arguments |  |
| :--- | :--- |
|  `username_or_email` |  (str) The username or email address of the user |

| Returns |  |
| :--- | :--- |
|  A `User` object or None if a user couldn't be found |

### `users`

[View source](https://www.github.com/wandb/wandb/tree/v0.16.6/wandb/apis/public/api.py#L727-L741)

```python
users(
    username_or_email
)
```

Return all users from a partial username or email address query.

Note: This function only works for Local Admins, if you are trying to get your own user object, please use `api.viewer`.

| Arguments |  |
| :--- | :--- |
|  `username_or_email` |  (str) The prefix or suffix of the user you want to find |

| Returns |  |
| :--- | :--- |
|  An array of `User` objects |

| Class Variables |  |
| :--- | :--- |
|  `CREATE_PROJECT`<a id="CREATE_PROJECT"></a> |   |
|  `USERS_QUERY`<a id="USERS_QUERY"></a> |   |
|  `VIEWER_QUERY`<a id="VIEWER_QUERY"></a> |   |
