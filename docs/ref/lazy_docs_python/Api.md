import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

# Api

<CTAButtons githubLink='https://github.com/wandb/wandb/blob/main/wandb/apis/public/api.py'/>




## <kbd>class</kbd> `Api`
Used for querying the wandb server. 



**Examples:**
  Most common way to initialize ``` wandb.Api()```



**Args:**


 - `    overrides`:  (dict) You can set `base_url` if you are using a wandb server

 - `        other than https`: //api.wandb.ai.
        You can also set defaults for `entity`, `project`, and `run`.


### <kbd>method</kbd> `Api.__init__`

```python
__init__(
    overrides: Optional[Dict[str, Any]] = None,
    timeout: Optional[int] = None,
    api_key: Optional[str] = None
) → None
```






---

#### <kbd>property</kbd> Api.api_key





---

#### <kbd>property</kbd> Api.client





---

#### <kbd>property</kbd> Api.default_entity





---

#### <kbd>property</kbd> Api.user_agent





---

#### <kbd>property</kbd> Api.viewer







---

### <kbd>method</kbd> `Api.artifact`

```python
artifact(name: str, type: Optional[str] = None)
```

Return a single artifact by parsing path in the form `project/name` or `entity/project/name`. 



**Args:**
 
 - `name`:  (str) An artifact name. May be prefixed with project/ or entity/project/.  If no entity is specified in the name, the Run or API setting's entity is used.  Valid names can be in the following forms: 
 - `name`: version 
 - `name`: alias 
 - `type`:  (str, optional) The type of artifact to fetch. 



**Returns:**
 An `Artifact` object. 



**Raises:**
 
 - `ValueError`:  If the artifact name is not specified. 
 - `ValueError`:  If the artifact type is specified but does not match the type of the fetched artifact. 



**Note:**

> This method is intended for external use only. Do not call `api.artifact()` within the wandb repository code. 

---

### <kbd>method</kbd> `Api.artifact_collection`

```python
artifact_collection(type_name: str, name: str) → public.ArtifactCollection
```

Return a single artifact collection by type and parsing path in the form `entity/project/name`. 



**Args:**
 
 - `type_name`:  (str) The type of artifact collection to fetch. 
 - `name`:  (str) An artifact collection name. May be prefixed with entity/project. 



**Returns:**
 An `ArtifactCollection` object. 

---

### <kbd>method</kbd> `Api.artifact_collection_exists`

```python
artifact_collection_exists(name: str, type: str) → bool
```

Return whether an artifact collection exists within a specified project and entity. 



**Args:**
 
 - `name`:  (str) An artifact collection name. May be prefixed with entity/project.  If entity or project is not specified, it will be inferred from the override params if populated.  Otherwise, entity will be pulled from the user settings and project will default to "uncategorized". 
 - `type`:  (str) The type of artifact collection 



**Returns:**
 True if the artifact collection exists, False otherwise. 

---

### <kbd>method</kbd> `Api.artifact_collections`

```python
artifact_collections(
    project_name: str,
    type_name: str,
    per_page: Optional[int] = 50
) → public.ArtifactCollections
```

Return a collection of matching artifact collections. 



**Args:**
 
 - `project_name`:  (str) The name of the project to filter on. 
 - `type_name`:  (str) The name of the artifact type to filter on. 
 - `per_page`:  (int, optional) Sets the page size for query pagination.  None will use the default size.  Usually there is no reason to change this. 



**Returns:**
 An iterable `ArtifactCollections` object. 

---

### <kbd>method</kbd> `Api.artifact_exists`

```python
artifact_exists(name: str, type: Optional[str] = None) → bool
```

Return whether an artifact version exists within a specified project and entity. 



**Args:**
 
 - `name`:  (str) An artifact name. May be prefixed with entity/project.  If entity or project is not specified, it will be inferred from the override params if populated.  Otherwise, entity will be pulled from the user settings and project will default to "uncategorized".  Valid names can be in the following forms: 
 - `name`: version 
 - `name`: alias 
 - `type`:  (str, optional) The type of artifact 



**Returns:**
 True if the artifact version exists, False otherwise. 

---

### <kbd>method</kbd> `Api.artifact_type`

```python
artifact_type(
    type_name: str,
    project: Optional[str] = None
) → public.ArtifactType
```

Return the matching `ArtifactType`. 



**Args:**
 
 - `type_name`:  (str) The name of the artifact type to retrieve. 
 - `project`:  (str, optional) If given, a project name or path to filter on. 



**Returns:**
 An `ArtifactType` object. 

---

### <kbd>method</kbd> `Api.artifact_types`

```python
artifact_types(project: Optional[str] = None) → public.ArtifactTypes
```

Return a collection of matching artifact types. 



**Args:**
 
 - `project`:  (str, optional) If given, a project name or path to filter on. 



**Returns:**
 An iterable `ArtifactTypes` object. 

---

### <kbd>method</kbd> `Api.artifact_versions`

```python
artifact_versions(type_name, name, per_page=50)
```

Deprecated, use `artifacts(type_name, name)` instead. 

---

### <kbd>method</kbd> `Api.artifacts`

```python
artifacts(
    type_name: str,
    name: str,
    per_page: Optional[int] = 50,
    tags: Optional[List[str]] = None
) → public.Artifacts
```

Return an `Artifacts` collection from the given parameters. 



**Args:**
 
 - `type_name`:  (str) The type of artifacts to fetch. 
 - `name`:  (str) An artifact collection name. May be prefixed with entity/project. 
 - `per_page`:  (int, optional) Sets the page size for query pagination.  None will use the default size.  Usually there is no reason to change this. 
 - `tags`:  (list[str], optional) Only return artifacts with all of these tags. 



**Returns:**
 An iterable `Artifacts` object. 

---

### <kbd>method</kbd> `Api.create_project`

```python
create_project(name: str, entity: str) → None
```

Create a new project. 



**Args:**
 
 - `name`:  (str) The name of the new project. 
 - `entity`:  (str) The entity of the new project. 

---

### <kbd>method</kbd> `Api.create_run`

```python
create_run(
    run_id: Optional[str] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None
) → public.Run
```

Create a new run. 



**Args:**
 
 - `run_id`:  (str, optional) The ID to assign to the run, if given.  The run ID is automatically generated by  default, so in general, you do not need to specify this and should only do so at your own risk. 
 - `project`:  (str, optional) If given, the project of the new run. 
 - `entity`:  (str, optional) If given, the entity of the new run. 



**Returns:**
 The newly created `Run`. 

---

### <kbd>method</kbd> `Api.create_run_queue`

```python
create_run_queue(
    name: str,
    type: 'public.RunQueueResourceType',
    entity: Optional[str] = None,
    prioritization_mode: Optional[ForwardRef('public.RunQueuePrioritizationMode')] = None,
    config: Optional[dict] = None,
    template_variables: Optional[dict] = None
) → public.RunQueue
```

Create a new run queue (launch). 



**Args:**
 
 - `name`:  (str) Name of the queue to create 
 - `type`:  (str) Type of resource to be used for the queue. One of "local-container", "local-process", "kubernetes", "sagemaker", or "gcp-vertex". 
 - `entity`:  (str) Optional name of the entity to create the queue. If None, will use the configured or default entity. 
 - `prioritization_mode`:  (str) Optional version of prioritization to use. Either "V0" or None 
 - `config`:  (dict) Optional default resource configuration to be used for the queue. Use handlebars (eg. `{{var}}`) to specify template variables. 
 - `template_variables`:  (dict) A dictionary of template variable schemas to be used with the config.



**Returns:**
 The newly created `RunQueue` 



**Raises:**
 ValueError if any of the parameters are invalid wandb.Error on wandb API errors 

---

### <kbd>method</kbd> `Api.create_team`

```python
create_team(team, admin_username=None)
```

Create a new team. 



**Args:**
 
 - `team`:  (str) The name of the team 
 - `admin_username`:  (str) optional username of the admin user of the team, defaults to the current user. 



**Returns:**
 A `Team` object 

---

### <kbd>method</kbd> `Api.create_user`

```python
create_user(email, admin=False)
```

Create a new user. 



**Args:**
 
 - `email`:  (str) The email address of the user 
 - `admin`:  (bool) Whether this user should be a global instance admin 



**Returns:**
 A `User` object 

---

### <kbd>method</kbd> `Api.flush`

```python
flush()
```

Flush the local cache. 

The api object keeps a local cache of runs, so if the state of the run may change while executing your script you must clear the local cache with `api.flush()` to get the latest values associated with the run. 

---

### <kbd>method</kbd> `Api.from_path`

```python
from_path(path)
```

Return a run, sweep, project or report from a path. 






**Args:**
 
 - `path`:  (str) The path to the project, run, sweep or report 



**Returns:**
 A `Project`, `Run`, `Sweep`, or `BetaReport` instance. 



**Raises:**
 wandb.Error if path is invalid or the object doesn't exist 

---

### <kbd>method</kbd> `Api.job`

```python
job(name: Optional[str], path: Optional[str] = None) → public.Job
```

Return a `Job` from the given parameters. 



**Args:**
 
 - `name`:  (str) The job name. 
 - `path`:  (str, optional) If given, the root path in which to download the job artifact. 



**Returns:**
 A `Job` object. 

---

### <kbd>method</kbd> `Api.list_jobs`

```python
list_jobs(entity: str, project: str) → List[Dict[str, Any]]
```

Return a list of jobs, if any, for the given entity and project. 



**Args:**
 
 - `entity`:  (str) The entity for the listed job(s). 
 - `project`:  (str) The project for the listed job(s). 



**Returns:**
 A list of matching jobs. 

---

### <kbd>method</kbd> `Api.project`

```python
project(name: str, entity: Optional[str] = None) → public.Project
```

Return the `Project` with the given name (and entity, if given). 



**Args:**
 
 - `name`:  (str) The project name. 
 - `entity`:  (str) Name of the entity requested.  If None, will fall back to the  default entity passed to `Api`.  If no default entity, will raise a `ValueError`. 



**Returns:**
 A `Project` object. 

---

### <kbd>method</kbd> `Api.projects`

```python
projects(
    entity: Optional[str] = None,
    per_page: Optional[int] = 200
) → public.Projects
```

Get projects for a given entity. 



**Args:**
 
 - `entity`:  (str) Name of the entity requested.  If None, will fall back to the  default entity passed to `Api`.  If no default entity, will raise a `ValueError`. 
 - `per_page`:  (int) Sets the page size for query pagination.  None will use the default size.  Usually there is no reason to change this. 



**Returns:**
 A `Projects` object which is an iterable collection of `Project` objects. 

---

### <kbd>method</kbd> `Api.queued_run`

```python
queued_run(
    entity,
    project,
    queue_name,
    run_queue_item_id,
    project_queue=None,
    priority=None
)
```

Return a single queued run based on the path. 

Parses paths of the form entity/project/queue_id/run_queue_item_id. 

---

### <kbd>method</kbd> `Api.reports`

```python
reports(
    path: str = '',
    name: Optional[str] = None,
    per_page: Optional[int] = 50
) → public.Reports
```

Get reports for a given project path. 

WARNING: This api is in beta and will likely change in a future release 



**Args:**
 
 - `path`:  (str) path to project the report resides in, should be in the form: "entity/project" 
 - `name`:  (str, optional) optional name of the report requested. 
 - `per_page`:  (int) Sets the page size for query pagination.  None will use the default size.  Usually there is no reason to change this. 



**Returns:**
 A `Reports` object which is an iterable collection of `BetaReport` objects. 

---

### <kbd>method</kbd> `Api.run`

```python
run(path='')
```

Return a single run by parsing path in the form entity/project/run_id. 



**Args:**
 
 - `path`:  (str) path to run in the form `entity/project/run_id`.  If `api.entity` is set, this can be in the form `project/run_id`  and if `api.project` is set this can just be the run_id. 



**Returns:**
 A `Run` object. 

---

### <kbd>method</kbd> `Api.run_queue`

```python
run_queue(entity, name)
```

Return the named `RunQueue` for entity. 

To create a new `RunQueue`, use `wandb.Api().create_run_queue(...)`. 

---

### <kbd>method</kbd> `Api.runs`

```python
runs(
    path: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    order: str = '+created_at',
    per_page: int = 50,
    include_sweeps: bool = True
)
```

Return a set of runs from a project that match the filters provided. 

You can filter by `config.*`, `summary_metrics.*`, `tags`, `state`, `entity`, `createdAt`, etc. 





**Args:**
 
 - `path`:  (str) path to project, should be in the form: "entity/project" 
 - `filters`:  (dict) queries for specific runs using the MongoDB query language.  You can filter by run properties such as config.key, summary_metrics.key, state, entity, createdAt, etc. 
 - `For example`:  `{"config.experiment_name": "foo"}` would find runs with a config entry  of experiment name set to "foo" You can compose operations to make more complicated queries, 
 - `see Reference for the language is at  https`: //docs.mongodb.com/manual/reference/operator/query 
 - `order`:  (str) Order can be `created_at`, `heartbeat_at`, `config.*.value`, or `summary_metrics.*`.  If you prepend order with a + order is ascending.  If you prepend order with a - order is descending (default).  The default order is run.created_at from oldest to newest. 
 - `per_page`:  (int) Sets the page size for query pagination. 
 - `include_sweeps`:  (bool) Whether to include the sweep runs in the results. 



**Returns:**
 A `Runs` object, which is an iterable collection of `Run` objects. 

---

### <kbd>method</kbd> `Api.sweep`

```python
sweep(path='')
```

Return a sweep by parsing path in the form `entity/project/sweep_id`. 



**Args:**
 
 - `path`:  (str, optional) path to sweep in the form entity/project/sweep_id.  If `api.entity`  is set, this can be in the form project/sweep_id and if `api.project` is set  this can just be the sweep_id. 



**Returns:**
 A `Sweep` object. 

---

### <kbd>method</kbd> `Api.sync_tensorboard`

```python
sync_tensorboard(root_dir, run_id=None, project=None, entity=None)
```

Sync a local directory containing tfevent files to wandb. 

---

### <kbd>method</kbd> `Api.team`

```python
team(team: str) → public.Team
```

Return the matching `Team` with the given name. 



**Args:**
 
 - `team`:  (str) The name of the team. 



**Returns:**
 A `Team` object. 

---

### <kbd>method</kbd> `Api.upsert_run_queue`

```python
upsert_run_queue(
    name: str,
    resource_config: dict,
    resource_type: 'public.RunQueueResourceType',
    entity: Optional[str] = None,
    template_variables: Optional[dict] = None,
    external_links: Optional[dict] = None,
    prioritization_mode: Optional[ForwardRef('public.RunQueuePrioritizationMode')] = None
)
```

Upsert a run queue (launch). 



**Args:**
 
 - `name`:  (str) Name of the queue to create 
 - `entity`:  (str) Optional name of the entity to create the queue. If None, will use the configured or default entity. 
 - `resource_config`:  (dict) Optional default resource configuration to be used for the queue. Use handlebars (eg. `{{var}}`) to specify template variables. 
 - `resource_type`:  (str) Type of resource to be used for the queue. One of "local-container", "local-process", "kubernetes", "sagemaker", or "gcp-vertex". 
 - `template_variables`:  (dict) A dictionary of template variable schemas to be used with the config. 



**Returns:**
 The upserted `RunQueue`. 



**Raises:**
 ValueError if any of the parameters are invalid wandb.Error on wandb API errors 

---

### <kbd>method</kbd> `Api.user`

```python
user(username_or_email: str) → Optional[ForwardRef('public.User')]
```

Return a user from a username or email address. 

Note: This function only works for Local Admins, if you are trying to get your own user object, please use `api.viewer`. 



**Args:**
 
 - `username_or_email`:  (str) The username or email address of the user 



**Returns:**
 A `User` object or None if a user couldn't be found 

---

### <kbd>method</kbd> `Api.users`

```python
users(username_or_email: str) → List[ForwardRef('public.User')]
```

Return all users from a partial username or email address query. 

Note: This function only works for Local Admins, if you are trying to get your own user object, please use `api.viewer`. 



**Args:**
 
 - `username_or_email`:  (str) The prefix or suffix of the user you want to find 



**Returns:**
 An array of `User` objects