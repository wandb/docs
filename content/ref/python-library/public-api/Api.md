---
title: api
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/apis/public/api.py >}}




# <kbd>module</kbd> `wandb.apis.public`
Use the Public API to export or update data that you have saved to W&B. 

Before using this API, you'll want to log data from your script — check the [Quickstart](https://docs.wandb.ai/quickstart) for more details. 

You might use the Public API to 
 - update metadata or metrics for an experiment after it has been completed, 
 - pull down your results as a dataframe for post-hoc analysis in a Jupyter notebook, or 
 - check your saved model artifacts for those tagged as `ready-to-deploy`. 

For more on using the Public API, check out [our guide](https://docs.wandb.com/guides/track/public-api-guide). 



## <kbd>class</kbd> `RetryingClient`




### <kbd>method</kbd> `RetryingClient.__init__`

```python
__init__(client: wandb_gql.client.Client)
```






---

### <kbd>property</kbd> RetryingClient.app_url





---

### <kbd>property</kbd> RetryingClient.server_info







---

### <kbd>method</kbd> `RetryingClient.execute`

```python
execute(*args, **kwargs)
```





---

### <kbd>method</kbd> `RetryingClient.version_supported`

```python
version_supported(min_version: str) → bool
```






---

## <kbd>class</kbd> `Api`
Used for querying the W&B server. 



**Examples:**
 ```python
import wandb

wandb.Api()
``` 

### <kbd>method</kbd> `Api.__init__`

```python
__init__(
    overrides: Optional[Dict[str, Any]] = None,
    timeout: Optional[int] = None,
    api_key: Optional[str] = None
) → None
```

Initialize the API. 



**Args:**
 
 - `overrides` (dict[str, Any] | None):  You can set `base_url` if you are 
 - `using a W&B server other than `https`: //api.wandb.ai`. You can also set defaults for `entity`, `project`, and `run`. 
 - `timeout` (int | None):  HTTP timeout in seconds for API requests. If not  specified, the default timeout will be used. 
 - `api_key` (str | None):  API key to use for authentication. If not provided,  the API key from the current environment or configuration will be used. 


---

### <kbd>property</kbd> Api.api_key

Returns W&B API key. 

---

### <kbd>property</kbd> Api.client

Returns the client object. 

---

### <kbd>property</kbd> Api.default_entity

Returns the default W&B entity. 

---

### <kbd>property</kbd> Api.user_agent

Returns W&B public user agent. 

---

### <kbd>property</kbd> Api.viewer

Returns the viewer object. 



---

### <kbd>method</kbd> `Api.artifact`

```python
artifact(name: str, type: Optional[str] = None)
```

Returns a single artifact. 



**Args:**
 
 - `name`:  The artifact's name. The name of an artifact resembles a  filepath that consists, at a minimum, the name of the project  the artifact was logged to, the name of the artifact, and the  artifact's version or alias. Optionally append the entity that  logged the artifact as a prefix followed by a forward slash.  If no entity is specified in the name, the Run or API  setting's entity is used. 
 - `type`:  The type of artifact to fetch. 



**Returns:**
 An `Artifact` object. 



**Raises:**
 
 - `ValueError`:  If the artifact name is not specified. 
 - `ValueError`:  If the artifact type is specified but does not  match the type of the fetched artifact. 



**Examples:**
 In the proceeding code snippets "entity", "project", "artifact", "version", and "alias" are placeholders for your W&B entity, name of the project the artifact is in, the name of the artifact, and artifact's version, respectively. 

```python
import wandb

# Specify the project, artifact's name, and the artifact's alias
wandb.Api().artifact(name="project/artifact:alias")

# Specify the project, artifact's name, and a specific artifact version
wandb.Api().artifact(name="project/artifact:version")

# Specify the entity, project, artifact's name, and the artifact's alias
wandb.Api().artifact(name="entity/project/artifact:alias")

# Specify the entity, project, artifact's name, and a specific artifact version
wandb.Api().artifact(name="entity/project/artifact:version")
``` 



**Note:**

> This method is intended for external use only. Do not call `api.artifact()` within the wandb repository code. 

---

### <kbd>method</kbd> `Api.artifact_collection`

```python
artifact_collection(type_name: str, name: str) → public.ArtifactCollection
```

Returns a single artifact collection by type. 

You can use the returned `ArtifactCollection` object to retrieve information about specific artifacts in that collection, and more. 



**Args:**
 
 - `type_name`:  The type of artifact collection to fetch. 
 - `name`:  An artifact collection name. Optionally append the entity  that logged the artifact as a prefix followed by a forward  slash. 



**Returns:**
 An `ArtifactCollection` object. 



**Examples:**
 In the proceeding code snippet "type", "entity", "project", and "artifact_name" are placeholders for the collection type, your W&B entity, name of the project the artifact is in, and the name of the artifact, respectively. 

```python
import wandb

collections = wandb.Api().artifact_collection(
    type_name="type", name="entity/project/artifact_name"
)

# Get the first artifact in the collection
artifact_example = collections.artifacts()[0]

# Download the contents of the artifact to the specified root directory.
artifact_example.download()
``` 

---

### <kbd>method</kbd> `Api.artifact_collection_exists`

```python
artifact_collection_exists(name: str, type: str) → bool
```

Whether an artifact collection exists within a specified project and entity. 



**Args:**
 
 - `name`:  An artifact collection name. Optionally append the  entity that logged the artifact as a prefix followed by  a forward slash. If entity or project is not specified,  infer the collection from the override params if they exist.  Otherwise, entity is pulled from the user settings and project  will default to "uncategorized". 
 - `type`:  The type of artifact collection. 



**Returns:**
 True if the artifact collection exists, False otherwise. 



**Examples:**
 In the proceeding code snippet "type", and "collection_name" refer to the type of the artifact collection and the name of the collection, respectively. 

```python
import wandb

wandb.Api.artifact_collection_exists(type="type", name="collection_name")
``` 

---

### <kbd>method</kbd> `Api.artifact_collections`

```python
artifact_collections(
    project_name: str,
    type_name: str,
    per_page: int = 50
) → public.ArtifactCollections
```

Returns a collection of matching artifact collections. 



**Args:**
 
 - `project_name`:  The name of the project to filter on. 
 - `type_name`:  The name of the artifact type to filter on. 
 - `per_page`:  Sets the page size for query pagination.  None will use the default size.  Usually there is no reason to change this. 



**Returns:**
 An iterable `ArtifactCollections` object. 

---

### <kbd>method</kbd> `Api.artifact_exists`

```python
artifact_exists(name: str, type: Optional[str] = None) → bool
```

Whether an artifact version exists within the specified project and entity. 



**Args:**
 
 - `name`:  The name of artifact. Add the artifact's entity and project  as a prefix. Append the version or the alias of the artifact  with a colon. If the entity or project is not specified,  W&B uses override parameters if populated. Otherwise, the  entity is pulled from the user settings and the project is  set to "Uncategorized". 
 - `type`:  The type of artifact. 



**Returns:**
 True if the artifact version exists, False otherwise. 



**Examples:**
 In the proceeding code snippets "entity", "project", "artifact", "version", and "alias" are placeholders for your W&B entity, name of the project the artifact is in, the name of the artifact, and artifact's version, respectively. 

```python
import wandb

wandb.Api().artifact_exists("entity/project/artifact:version")
wandb.Api().artifact_exists("entity/project/artifact:alias")
``` 

---

### <kbd>method</kbd> `Api.artifact_type`

```python
artifact_type(
    type_name: str,
    project: Optional[str] = None
) → public.ArtifactType
```

Returns the matching `ArtifactType`. 



**Args:**
 
 - `type_name`:  The name of the artifact type to retrieve. 
 - `project`:  If given, a project name or path to filter on. 



**Returns:**
 An `ArtifactType` object. 

---

### <kbd>method</kbd> `Api.artifact_types`

```python
artifact_types(project: Optional[str] = None) → public.ArtifactTypes
```

Returns a collection of matching artifact types. 



**Args:**
 
 - `project`:  The project name or path to filter on. 



**Returns:**
 An iterable `ArtifactTypes` object. 

---

### <kbd>method</kbd> `Api.artifact_versions`

```python
artifact_versions(type_name, name, per_page=50)
```

Deprecated. Use `Api.artifacts(type_name, name)` method instead. 

---

### <kbd>method</kbd> `Api.artifacts`

```python
artifacts(
    type_name: str,
    name: str,
    per_page: int = 50,
    tags: Optional[List[str]] = None
) → public.Artifacts
```

Return an `Artifacts` collection. 



**Args:**
 type_name: The type of artifacts to fetch. name: The artifact's collection name. Optionally append the  entity that logged the artifact as a prefix followed by  a forward slash. per_page: Sets the page size for query pagination. If set to  `None`, use the default size. Usually there is no reason  to change this. tags: Only return artifacts with all of these tags. 



**Returns:**
  An iterable `Artifacts` object. 



**Examples:**
 In the proceeding code snippet, "type", "entity", "project", and "artifact_name" are placeholders for the artifact type, W&B entity, name of the project the artifact was logged to, and the name of the artifact, respectively. 

```python
import wandb

wandb.Api().artifacts(type_name="type", name="entity/project/artifact_name")
``` 

---

### <kbd>method</kbd> `Api.automation`

```python
automation(name: str, entity: Optional[str] = None) → Automation
```

Returns the only Automation matching the parameters. 



**Args:**
 
 - `name`:  The name of the automation to fetch. 
 - `entity`:  The entity to fetch the automation for. 



**Raises:**
 
 - `ValueError`:  If zero or multiple Automations match the search criteria. 



**Examples:**
 Get an existing automation named "my-automation": 

```python
    import wandb

    api = wandb.Api()
    automation = api.automation(name="my-automation")
    ``` 

Get an existing automation named "other-automation", from the entity "my-team": 

```python
    automation = api.automation(name="other-automation", entity="my-team")
    ``` 

---

### <kbd>method</kbd> `Api.automations`

```python
automations(
    entity: Optional[str] = None,
    name: Optional[str] = None,
    per_page: int = 50
) → Iterator[ForwardRef('Automation')]
```

Returns an iterator over all Automations that match the given parameters. 

If no parameters are provided, the returned iterator will contain all Automations that the user has access to. 



**Args:**
 
 - `entity`:  The entity to fetch the automations for. 
 - `name`:  The name of the automation to fetch. 
 - `per_page`:  The number of automations to fetch per page.  Defaults to 50.  Usually there is no reason to change this. 



**Returns:**
 A list of automations. 



**Examples:**
 Fetch all existing automations for the entity "my-team": 

```python
    import wandb

    api = wandb.Api()
    automations = api.automations(entity="my-team")
    ``` 

---

### <kbd>method</kbd> `Api.create_automation`

```python
create_automation(
    obj: 'NewAutomation',
    fetch_existing: bool = False,
    **kwargs: typing_extensions.Unpack[ForwardRef('WriteAutomationsKwargs')]
) → Automation
```

Create a new Automation. 



**Args:**
  obj:  The automation to create.  fetch_existing:  If True, and a conflicting automation already exists, attempt  to fetch the existing automation instead of raising an error.  **kwargs:  Any additional values to assign to the automation before  creating it.  If given, these will override any values that may  already be set on the automation: 
        - `name`: The name of the automation. 
        - `description`: The description of the automation. 
        - `enabled`: Whether the automation is enabled. 
        - `scope`: The scope of the automation. 
        - `event`: The event that triggers the automation. 
        - `action`: The action that is triggered by the automation. 



**Returns:**
  The saved Automation. 



**Examples:**
  Create a new automation named "my-automation" that sends a Slack notification  when a run within a specific project logs a metric exceeding a custom threshold: 

```python
     import wandb
     from wandb.automations import OnRunMetric, RunEvent, SendNotification

     api = wandb.Api()

     project = api.project("my-project", entity="my-team")

     # Use the first Slack integration for the team
     slack_hook = next(api.slack_integrations(entity="my-team"))

     event = OnRunMetric(
         scope=project,
         filter=RunEvent.metric("custom-metric") > 10,
     )
     action = SendNotification.from_integration(slack_hook)

     automation = api.create_automation(
         event >> action,
         name="my-automation",
         description="Send a Slack message whenever 'custom-metric' exceeds 10.",
     )
    ``` 

---

### <kbd>method</kbd> `Api.create_project`

```python
create_project(name: str, entity: str) → None
```

Create a new project. 



**Args:**
 
 - `name`:  The name of the new project. 
 - `entity`:  The entity of the new project. 

---

### <kbd>method</kbd> `Api.create_registry`

```python
create_registry(
    name: str,
    visibility: Literal['organization', 'restricted'],
    organization: Optional[str] = None,
    description: Optional[str] = None,
    artifact_types: Optional[List[str]] = None
) → Registry
```

Create a new registry. 



**Args:**
 
 - `name`:  The name of the registry. Name must be unique within the organization. 
 - `visibility`:  The visibility of the registry. 
 - `organization`:  Anyone in the organization can view this registry. You can  edit their roles later from the settings in the UI. 
 - `restricted`:  Only invited members via the UI can access this registry.  Public sharing is disabled. 
 - `organization`:  The organization of the registry.  If no organization is set in the settings, the organization will be  fetched from the entity if the entity only belongs to one organization. 
 - `description`:  The description of the registry. 
 - `artifact_types`:  The accepted artifact types of the registry. A type is no 
 - `more than 128 characters and do not include characters `/` or ``: `. If not specified, all types are accepted. Allowed types added to the registry cannot be removed later. 



**Returns:**
 A registry object. 



**Examples:**
 ```python
    import wandb

    api = wandb.Api()
    registry = api.create_registry(
        name="my-registry",
        visibility="restricted",
        organization="my-org",
        description="This is a test registry",
        artifact_types=["model"],
    )
    ``` 

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
 
 - `run_id`:  The ID to assign to the run. If not specified, W&B  creates a random ID. 
 - `project`:  The project where to log the run to. If no project is specified,  log the run to a project called "Uncategorized". 
 - `entity`:  The entity that owns the project. If no entity is  specified, log the run to the default entity. 



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

Create a new run queue in W&B Launch. 



**Args:**
 
 - `name`:  Name of the queue to create 
 - `type`:  Type of resource to be used for the queue. One of  "local-container", "local-process", "kubernetes","sagemaker",  or "gcp-vertex". 
 - `entity`:  Name of the entity to create the queue. If `None`, use  the configured or default entity. 
 - `prioritization_mode`:  Version of prioritization to use.  Either "V0" or `None`. 
 - `config`:  Default resource configuration to be used for the queue.  Use handlebars (eg. `{{var}}`) to specify template variables. 
 - `template_variables`:  A dictionary of template variable schemas to  use with the config. 



**Returns:**
 The newly created `RunQueue`. 



**Raises:**
 `ValueError` if any of the parameters are invalid `wandb.Error` on wandb API errors 

---

### <kbd>method</kbd> `Api.create_team`

```python
create_team(team: str, admin_username: Optional[str] = None) → public.Team
```

Create a new team. 



**Args:**
 
 - `team`:  The name of the team 
 - `admin_username`:  Username of the admin user of the team.  Defaults to the current user. 



**Returns:**
 A `Team` object. 

---

### <kbd>method</kbd> `Api.create_user`

```python
create_user(email: str, admin: Optional[bool] = False)
```

Create a new user. 



**Args:**
 
 - `email`:  The email address of the user. 
 - `admin`:  Set user as a global instance administrator. 



**Returns:**
 A `User` object. 

---

### <kbd>method</kbd> `Api.delete_automation`

```python
delete_automation(obj: Union[ForwardRef('Automation'), str]) → Literal[True]
```

Delete an automation. 



**Args:**
 
 - `obj`:  The automation to delete, or its ID. 



**Returns:**
 True if the automation was deleted successfully. 

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
from_path(path: str)
```

Return a run, sweep, project or report from a path. 



**Args:**
 
 - `path`:  The path to the project, run, sweep or report 



**Returns:**
 A `Project`, `Run`, `Sweep`, or `BetaReport` instance. 



**Raises:**
 `wandb.Error` if path is invalid or the object doesn't exist. 



**Examples:**
 In the proceeding code snippets "project", "team", "run_id", "sweep_id", and "report_name" are placeholders for the project, team, run ID, sweep ID, and the name of a specific report, respectively. 

```python
import wandb

api = wandb.Api()

project = api.from_path("project")
team_project = api.from_path("team/project")
run = api.from_path("team/project/runs/run_id")
sweep = api.from_path("team/project/sweeps/sweep_id")
report = api.from_path("team/project/reports/report_name")
``` 

---

### <kbd>method</kbd> `Api.integrations`

```python
integrations(
    entity: Optional[str] = None,
    per_page: int = 50
) → Iterator[ForwardRef('Integration')]
```

Return an iterator of all integrations for an entity. 



**Args:**
 
 - `entity`:  The entity (e.g. team name) for which to  fetch integrations.  If not provided, the user's default entity  will be used. 
 - `per_page`:  Number of integrations to fetch per page.  Defaults to 50.  Usually there is no reason to change this. 



**Yields:**
 
 - `Iterator[SlackIntegration | WebhookIntegration]`:  An iterator of any supported integrations. 

---

### <kbd>method</kbd> `Api.job`

```python
job(name: Optional[str], path: Optional[str] = None) → public.Job
```

Return a `Job` object. 



**Args:**
 
 - `name`:  The name of the job. 
 - `path`:  The root path to download the job artifact. 



**Returns:**
 A `Job` object. 

---

### <kbd>method</kbd> `Api.list_jobs`

```python
list_jobs(entity: str, project: str) → List[Dict[str, Any]]
```

Return a list of jobs, if any, for the given entity and project. 



**Args:**
 
 - `entity`:  The entity for the listed jobs. 
 - `project`:  The project for the listed jobs. 



**Returns:**
 A list of matching jobs. 

---

### <kbd>method</kbd> `Api.project`

```python
project(name: str, entity: Optional[str] = None) → public.Project
```

Return the `Project` with the given name (and entity, if given). 



**Args:**
 
 - `name`:  The project name. 
 - `entity`:  Name of the entity requested.  If None, will fall back to the  default entity passed to `Api`.  If no default entity, will  raise a `ValueError`. 



**Returns:**
 A `Project` object. 

---

### <kbd>method</kbd> `Api.projects`

```python
projects(entity: Optional[str] = None, per_page: int = 200) → public.Projects
```

Get projects for a given entity. 



**Args:**
 
 - `entity`:  Name of the entity requested.  If None, will fall back to  the default entity passed to `Api`.  If no default entity,  will raise a `ValueError`. 
 - `per_page`:  Sets the page size for query pagination. If set to `None`,  use the default size. Usually there is no reason to change this. 



**Returns:**
 A `Projects` object which is an iterable collection of `Project`objects. 

---

### <kbd>method</kbd> `Api.queued_run`

```python
queued_run(
    entity: str,
    project: str,
    queue_name: str,
    run_queue_item_id: str,
    project_queue=None,
    priority=None
)
```

Return a single queued run based on the path. 

Parses paths of the form `entity/project/queue_id/run_queue_item_id`. 

---

### <kbd>method</kbd> `Api.registries`

```python
registries(
    organization: Optional[str] = None,
    filter: Optional[Dict[str, Any]] = None
) → Registries
```

Returns a Registry iterator. 

Use the iterator to search and filter registries, collections, or artifact versions across your organization's registry. 



**Examples:**
  Find all registries with the names that contain "model" ```python
     import wandb

     api = wandb.Api()  # specify an org if your entity belongs to multiple orgs
     api.registries(filter={"name": {"$regex": "model"}})
    ``` 

 Find all collections in the registries with the name "my_collection" and the tag "my_tag" ```python
     api.registries().collections(filter={"name": "my_collection", "tag": "my_tag"})
    ``` 

 Find all artifact versions in the registries with a collection name that contains "my_collection" and a version that has the alias "best" ```python
     api.registries().collections(
         filter={"name": {"$regex": "my_collection"}}
     ).versions(filter={"alias": "best"})
    ``` 

 Find all artifact versions in the registries that contain "model" and have the tag "prod" or alias "best" ```python
     api.registries(filter={"name": {"$regex": "model"}}).versions(
         filter={"$or": [{"tag": "prod"}, {"alias": "best"}]}
     )
    ``` 



**Args:**
 
 - `organization`:  (str, optional) The organization of the registry to fetch.  If not specified, use the organization specified in the user's settings. 
 - `filter`:  (dict, optional) MongoDB-style filter to apply to each object in the registry iterator.  Fields available to filter for collections are  `name`, `description`, `created_at`, `updated_at`.  Fields available to filter for collections are  `name`, `tag`, `description`, `created_at`, `updated_at`  Fields available to filter for versions are  `tag`, `alias`, `created_at`, `updated_at`, `metadata` 



**Returns:**
 A registry iterator. 

---

### <kbd>method</kbd> `Api.registry`

```python
registry(name: str, organization: Optional[str] = None) → Registry
```

Return a registry given a registry name. 



**Args:**
 
 - `name`:  The name of the registry. This is without the `wandb-registry-`  prefix. 
 - `organization`:  The organization of the registry.  If no organization is set in the settings, the organization will be  fetched from the entity if the entity only belongs to one  organization. 



**Returns:**
 A registry object. 



**Examples:**
 Fetch and update a registry ```python
    import wandb

    api = wandb.Api()
    registry = api.registry(name="my-registry", organization="my-org")
    registry.description = "This is an updated description"
    registry.save()
    ``` 

---

### <kbd>method</kbd> `Api.reports`

```python
reports(
    path: str = '',
    name: Optional[str] = None,
    per_page: int = 50
) → public.Reports
```

Get reports for a given project path. 

Note: `wandb.Api.reports()` API is in beta and will likely change in future releases. 



**Args:**
 
 - `path`:  The path to project the report resides in. Specify the  entity that created the project as a prefix followed by a  forward slash. 
 - `name`:  Name of the report requested. 
 - `per_page`:  Sets the page size for query pagination. If set to  `None`, use the default size. Usually there is no reason to  change this. 



**Returns:**
 A `Reports` object which is an iterable collection of  `BetaReport` objects. 



**Examples:**
 ```python
import wandb

wandb.Api.reports("entity/project")
``` 

---

### <kbd>method</kbd> `Api.run`

```python
run(path='')
```

Return a single run by parsing path in the form `entity/project/run_id`. 



**Args:**
 
 - `path`:  Path to run in the form `entity/project/run_id`.  If `api.entity` is set, this can be in the form `project/run_id`  and if `api.project` is set this can just be the run_id. 



**Returns:**
 A `Run` object. 

---

### <kbd>method</kbd> `Api.run_queue`

```python
run_queue(entity: str, name: str)
```

Return the named `RunQueue` for entity. 

See `Api.create_run_queue` for more information on how to create a run queue. 

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

Fields you can filter by include: 
- `createdAt`: The timestamp when the run was created. (in ISO 8601 format, e.g. "2023-01-01T12:00:00Z") 
- `displayName`: The human-readable display name of the run. (e.g. "eager-fox-1") 
- `duration`: The total runtime of the run in seconds. 
- `group`: The group name used to organize related runs together. 
- `host`: The hostname where the run was executed. 
- `jobType`: The type of job or purpose of the run. 
- `name`: The unique identifier of the run. (e.g. "a1b2cdef") 
- `state`: The current state of the run. 
- `tags`: The tags associated with the run. 
- `username`: The username of the user who initiated the run 

Additionally, you can filter by items in the run config or summary metrics. Such as `config.experiment_name`, `summary_metrics.loss`, etc. 

For more complex filtering, you can use MongoDB query operators. For details, see: https://docs.mongodb.com/manual/reference/operator/query The following operations are supported: 
- `$and` 
- `$or` 
- `$nor` 
- `$eq` 
- `$ne` 
- `$gt` 
- `$gte` 
- `$lt` 
- `$lte` 
- `$in` 
- `$nin` 
- `$exists` 
- `$regex` 







**Args:**
 
 - `path`:  (str) path to project, should be in the form: "entity/project" 
 - `filters`:  (dict) queries for specific runs using the MongoDB query language.  You can filter by run properties such as config.key, summary_metrics.key, state, entity, createdAt, etc. 
 - `For example`:  `{"config.experiment_name": "foo"}` would find runs with a config entry  of experiment name set to "foo" 
 - `order`:  (str) Order can be `created_at`, `heartbeat_at`, `config.*.value`, or `summary_metrics.*`.  If you prepend order with a + order is ascending.  If you prepend order with a - order is descending (default).  The default order is run.created_at from oldest to newest. 
 - `per_page`:  (int) Sets the page size for query pagination. 
 - `include_sweeps`:  (bool) Whether to include the sweep runs in the results. 



**Returns:**
 A `Runs` object, which is an iterable collection of `Run` objects. 



**Examples:**
 ```python
# Find runs in project where config.experiment_name has been set to "foo"
api.runs(path="my_entity/project", filters={"config.experiment_name": "foo"})
``` 

```python
# Find runs in project where config.experiment_name has been set to "foo" or "bar"
api.runs(
    path="my_entity/project",
    filters={
         "$or": [
             {"config.experiment_name": "foo"},
             {"config.experiment_name": "bar"},
         ]
    },
)
``` 

```python
# Find runs in project where config.experiment_name matches a regex
# (anchors are not supported)
api.runs(
    path="my_entity/project",
    filters={"config.experiment_name": {"$regex": "b.*"}},
)
``` 

```python
# Find runs in project where the run name matches a regex
# (anchors are not supported)
api.runs(
    path="my_entity/project", filters={"display_name": {"$regex": "^foo.*"}}
)
``` 

```python
# Find runs in project sorted by ascending loss
api.runs(path="my_entity/project", order="+summary_metrics.loss")
``` 

---

### <kbd>method</kbd> `Api.slack_integrations`

```python
slack_integrations(
    entity: Optional[str] = None,
    per_page: int = 50
) → Iterator[ForwardRef('SlackIntegration')]
```

Returns an iterator of Slack integrations for an entity. 



**Args:**
 
 - `entity`:  The entity (e.g. team name) for which to  fetch integrations.  If not provided, the user's default entity  will be used. 
 - `per_page`:  Number of integrations to fetch per page.  Defaults to 50.  Usually there is no reason to change this. 



**Yields:**
 
 - `Iterator[SlackIntegration]`:  An iterator of Slack integrations. 



**Examples:**
 Get all registered Slack integrations for the team "my-team": ```python
    import wandb

    api = wandb.Api()
    slack_integrations = api.slack_integrations(entity="my-team")
    ``` 

Find only Slack integrations that post to channel names starting with "team-alerts-": ```python
    slack_integrations = api.slack_integrations(entity="my-team")
    team_alert_integrations = [
         ig
         for ig in slack_integrations
         if ig.channel_name.startswith("team-alerts-")
    ]
    ``` 

---

### <kbd>method</kbd> `Api.sweep`

```python
sweep(path='')
```

Return a sweep by parsing path in the form `entity/project/sweep_id`. 



**Args:**
 
 - `path`:  Path to sweep in the form entity/project/sweep_id.  If `api.entity` is set, this can be in the form  project/sweep_id and if `api.project` is set  this can just be the sweep_id. 



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
 
 - `team`:  The name of the team. 



**Returns:**
 A `Team` object. 

---

### <kbd>method</kbd> `Api.update_automation`

```python
update_automation(
    obj: 'Automation',
    create_missing: bool = False,
    **kwargs: typing_extensions.Unpack[ForwardRef('WriteAutomationsKwargs')]
) → Automation
```

Update an existing automation. 



**Args:**
 
 - `obj`:  The automation to update.  Must be an existing automation. create_missing (bool):  If True, and the automation does not exist, create it. **kwargs:  Any additional values to assign to the automation before  updating it.  If given, these will override any values that may  already be set on the automation: 
        - `name`: The name of the automation. 
        - `description`: The description of the automation. 
        - `enabled`: Whether the automation is enabled. 
        - `scope`: The scope of the automation. 
        - `event`: The event that triggers the automation. 
        - `action`: The action that is triggered by the automation. 



**Returns:**
 The updated automation. 



**Examples:**
 Disable and edit the description of an existing automation ("my-automation"): 

```python
    import wandb

    api = wandb.Api()

    automation = api.automation(name="my-automation")
    automation.enabled = False
    automation.description = "Kept for reference, but no longer used."

    updated_automation = api.update_automation(automation)
    ``` 

OR: 

```python
    import wandb

    api = wandb.Api()

    automation = api.automation(name="my-automation")

    updated_automation = api.update_automation(
         automation,
         enabled=False,
         description="Kept for reference, but no longer used.",
    )
    ``` 

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

Upsert a run queue in W&B Launch. 



**Args:**
 
 - `name`:  Name of the queue to create 
 - `entity`:  Optional name of the entity to create the queue. If `None`,  use the configured or default entity. 
 - `resource_config`:  Optional default resource configuration to be used  for the queue. Use handlebars (eg. `{{var}}`) to specify  template variables. 
 - `resource_type`:  Type of resource to be used for the queue. One of  "local-container", "local-process", "kubernetes", "sagemaker",  or "gcp-vertex". 
 - `template_variables`:  A dictionary of template variable schemas to  be used with the config. 
 - `external_links`:  Optional dictionary of external links to be used  with the queue. 
 - `prioritization_mode`:  Optional version of prioritization to use.  Either "V0" or None 



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

This function only works for local administrators. Use `api.viewer`  to get your own user object. 



**Args:**
 
 - `username_or_email`:  The username or email address of the user. 



**Returns:**
 A `User` object or None if a user is not found. 

---

### <kbd>method</kbd> `Api.users`

```python
users(username_or_email: str) → List[ForwardRef('public.User')]
```

Return all users from a partial username or email address query. 

This function only works for local administrators. Use `api.viewer`  to get your own user object. 



**Args:**
 
 - `username_or_email`:  The prefix or suffix of the user you want to find. 



**Returns:**
 An array of `User` objects. 

---

### <kbd>method</kbd> `Api.webhook_integrations`

```python
webhook_integrations(
    entity: Optional[str] = None,
    per_page: int = 50
) → Iterator[ForwardRef('WebhookIntegration')]
```

Returns an iterator of webhook integrations for an entity. 



**Args:**
 
 - `entity`:  The entity (e.g. team name) for which to  fetch integrations.  If not provided, the user's default entity  will be used. 
 - `per_page`:  Number of integrations to fetch per page.  Defaults to 50.  Usually there is no reason to change this. 



**Yields:**
 
 - `Iterator[WebhookIntegration]`:  An iterator of webhook integrations. 



**Examples:**
 Get all registered webhook integrations for the team "my-team": ```python
    import wandb

    api = wandb.Api()
    webhook_integrations = api.webhook_integrations(entity="my-team")
    ``` 

Find only webhook integrations that post requests to "https://my-fake-url.com": ```python
    webhook_integrations = api.webhook_integrations(entity="my-team")
    my_webhooks = [
         ig
         for ig in webhook_integrations
         if ig.url_endpoint.startswith("https://my-fake-url.com")
    ]
    ``` 


