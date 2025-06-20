---
title: Api
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L137-L2203 >}}

Used for querying the wandb server.

```python
Api(
    overrides: Optional[Dict[str, Any]] = None,
    timeout: Optional[int] = None,
    api_key: Optional[str] = None
) -> None
```

#### Examples:

Most common way to initialize

```
>>> wandb.Api()
```

| Args |  |
| :--- | :--- |
|  `overrides` |  (dict) You can set `base_url` if you are using a wandb server other than https://api.wandb.ai. You can also set defaults for `entity`, `project`, and `run`. |

| Attributes |  |
| :--- | :--- |

## Methods

### `artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L1332-L1354)

```python
artifact(
    name: str,
    type: Optional[str] = None
)
```

Return a single artifact by parsing path in the form `project/name` or `entity/project/name`.

| Args |  |
| :--- | :--- |
|  `name` |  (str) An artifact name. May be prefixed with project/ or entity/project/. If no entity is specified in the name, the Run or API setting's entity is used. Valid names can be in the following forms: name:version name:alias |
|  `type` |  (str, optional) The type of artifact to fetch. |

| Returns |  |
| :--- | :--- |
|  An `Artifact` object. |

| Raises |  |
| :--- | :--- |
|  `ValueError` |  If the artifact name is not specified. |
|  `ValueError` |  If the artifact type is specified but does not match the type of the fetched artifact. |

#### Note:

This method is intended for external use only. Do not call `api.artifact()` within the wandb repository code.

### `artifact_collection`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L1211-L1240)

```python
artifact_collection(
    type_name: str,
    name: str
) -> "public.ArtifactCollection"
```

Return a single artifact collection by type and parsing path in the form `entity/project/name`.

| Args |  |
| :--- | :--- |
|  `type_name` |  (str) The type of artifact collection to fetch. |
|  `name` |  (str) An artifact collection name. May be prefixed with entity/project. |

| Returns |  |
| :--- | :--- |
|  An `ArtifactCollection` object. |

### `artifact_collection_exists`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L1476-L1494)

```python
artifact_collection_exists(
    name: str,
    type: str
) -> bool
```

Return whether an artifact collection exists within a specified project and entity.

| Args |  |
| :--- | :--- |
|  `name` |  (str) An artifact collection name. May be prefixed with entity/project. If entity or project is not specified, it will be inferred from the override params if populated. Otherwise, entity will be pulled from the user settings and project will default to "uncategorized". |
|  `type` |  (str) The type of artifact collection |

| Returns |  |
| :--- | :--- |
|  True if the artifact collection exists, False otherwise. |

### `artifact_collections`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L1185-L1209)

```python
artifact_collections(
    project_name: str,
    type_name: str,
    per_page: int = 50
) -> "public.ArtifactCollections"
```

Return a collection of matching artifact collections.

| Args |  |
| :--- | :--- |
|  `project_name` |  (str) The name of the project to filter on. |
|  `type_name` |  (str) The name of the artifact type to filter on. |
|  `per_page` |  (int) Sets the page size for query pagination. Usually there is no reason to change this. |

| Returns |  |
| :--- | :--- |
|  An iterable `ArtifactCollections` object. |

### `artifact_exists`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L1453-L1474)

```python
artifact_exists(
    name: str,
    type: Optional[str] = None
) -> bool
```

Return whether an artifact version exists within a specified project and entity.

| Args |  |
| :--- | :--- |
|  `name` |  (str) An artifact name. May be prefixed with entity/project. If entity or project is not specified, it will be inferred from the override params if populated. Otherwise, entity will be pulled from the user settings and project will default to "uncategorized". Valid names can be in the following forms: name:version name:alias |
|  `type` |  (str, optional) The type of artifact |

| Returns |  |
| :--- | :--- |
|  True if the artifact version exists, False otherwise. |

### `artifact_type`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L1161-L1183)

```python
artifact_type(
    type_name: str,
    project: Optional[str] = None
) -> "public.ArtifactType"
```

Return the matching `ArtifactType`.

| Args |  |
| :--- | :--- |
|  `type_name` |  (str) The name of the artifact type to retrieve. |
|  `project` |  (str, optional) If given, a project name or path to filter on. |

| Returns |  |
| :--- | :--- |
|  An `ArtifactType` object. |

### `artifact_types`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L1140-L1159)

```python
artifact_types(
    project: Optional[str] = None
) -> "public.ArtifactTypes"
```

Return a collection of matching artifact types.

| Args |  |
| :--- | :--- |
|  `project` |  (str, optional) If given, a project name or path to filter on. |

| Returns |  |
| :--- | :--- |
|  An iterable `ArtifactTypes` object. |

### `artifact_versions`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L1242-L1252)

```python
artifact_versions(
    type_name, name, per_page=50
)
```

Deprecated, use `artifacts(type_name, name)` instead.

### `artifacts`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L1254-L1289)

```python
artifacts(
    type_name: str,
    name: str,
    per_page: int = 50,
    tags: Optional[List[str]] = None
) -> "public.Artifacts"
```

Return an `Artifacts` collection from the given parameters.

| Args |  |
| :--- | :--- |
|  `type_name` |  (str) The type of artifacts to fetch. |
|  `name` |  (str) An artifact collection name. May be prefixed with entity/project. |
|  `per_page` |  (int) Sets the page size for query pagination. Usually there is no reason to change this. |
|  `tags` |  (list[str], optional) Only return artifacts with all of these tags. |

| Returns |  |
| :--- | :--- |
|  An iterable `Artifacts` object. |

### `automation`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L1849-L1884)

```python
automation(
    name: str,
    *,
    entity: Optional[str] = None
) -> "Automation"
```

Returns the only Automation matching the parameters.

| Args |  |
| :--- | :--- |
|  `name` |  The name of the automation to fetch. |
|  `entity` |  The entity to fetch the automation for. |

| Raises |  |
| :--- | :--- |
|  `ValueError` |  If zero or multiple Automations match the search criteria. |

#### Examples:

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

### `automations`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L1886-L1940)

```python
automations(
    entity: Optional[str] = None,
    *,
    name: Optional[str] = None,
    per_page: int = 50
) -> Iterator['Automation']
```

Returns an iterator over all Automations that match the given parameters.

If no parameters are provided, the returned iterator will contain all
Automations that the user has access to.

| Args |  |
| :--- | :--- |
|  `entity` |  The entity to fetch the automations for. |
|  `name` |  The name of the automation to fetch. |
|  `per_page` |  The number of automations to fetch per page. Defaults to 50. Usually there is no reason to change this. |

| Returns |  |
| :--- | :--- |
|  A list of automations. |

#### Examples:

Fetch all existing automations for the entity "my-team":

```python
import wandb

api = wandb.Api()
automations = api.automations(entity="my-team")
```

### `create_automation`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L1942-L2047)

```python
create_automation(
    obj: "NewAutomation",
    *,
    fetch_existing: bool = (False),
    **kwargs
) -> "Automation"
```

Create a new Automation.

| Args |  |
| :--- | :--- |
|  `obj` |  The automation to create. |
|  `fetch_existing` |  If True, and a conflicting automation already exists, attempt to fetch the existing automation instead of raising an error. |
|  `**kwargs` |  Any additional values to assign to the automation before creating it. If given, these will override any values that may already be set on the automation: - `name`: The name of the automation. - `description`: The description of the automation. - `enabled`: Whether the automation is enabled. - `scope`: The scope of the automation. - `event`: The event that triggers the automation. - `action`: The action that is triggered by the automation. |

| Returns |  |
| :--- | :--- |
|  The saved Automation. |

#### Examples:

Create a new automation named "my-automation" that sends a Slack notification
when a run within a specific project logs a metric exceeding a custom threshold:

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

### `create_project`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L327-L334)

```python
create_project(
    name: str,
    entity: str
) -> None
```

Create a new project.

| Args |  |
| :--- | :--- |
|  `name` |  (str) The name of the new project. |
|  `entity` |  (str) The entity of the new project. |

### `create_registry`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L1597-L1669)

```python
create_registry(
    name: str,
    visibility: Literal['organization', 'restricted'],
    organization: Optional[str] = None,
    description: Optional[str] = None,
    artifact_types: Optional[List[str]] = None
) -> Registry
```

Create a new registry.

| Args |  |
| :--- | :--- |
|  `name` |  The name of the registry. Name must be unique within the organization. |
|  `visibility` |  The visibility of the registry. organization: Anyone in the organization can view this registry. You can edit their roles later from the settings in the UI. restricted: Only invited members via the UI can access this registry. Public sharing is disabled. |
|  `organization` |  The organization of the registry. If no organization is set in the settings, the organization will be fetched from the entity if the entity only belongs to one organization. |
|  `description` |  The description of the registry. |
|  `artifact_types` |  The accepted artifact types of the registry. A type is no more than 128 characters and do not include characters `/` or `:`. If not specified, all types are accepted. Allowed types added to the registry cannot be removed later. |

| Returns |  |
| :--- | :--- |
|  A registry object. |

#### Examples:

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

### `create_run`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L336-L356)

```python
create_run(
    *,
    run_id: Optional[str] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None
) -> "public.Run"
```

Create a new run.

| Args |  |
| :--- | :--- |
|  `run_id` |  (str, optional) The ID to assign to the run, if given. The run ID is automatically generated by default, so in general, you do not need to specify this and should only do so at your own risk. |
|  `project` |  (str, optional) If given, the project of the new run. |
|  `entity` |  (str, optional) If given, the entity of the new run. |

| Returns |  |
| :--- | :--- |
|  The newly created `Run`. |

### `create_run_queue`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L358-L468)

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

| Args |  |
| :--- | :--- |
|  `name` |  (str) Name of the queue to create |
|  `type` |  (str) Type of resource to be used for the queue. One of "local-container", "local-process", "kubernetes", "sagemaker", or "gcp-vertex". |
|  `entity` |  (str) Optional name of the entity to create the queue. If None, will use the configured or default entity. |
|  `prioritization_mode` |  (str) Optional version of prioritization to use. Either "V0" or None |
|  `config` |  (dict) Optional default resource configuration to be used for the queue. Use handlebars (eg. `{{var}}`) to specify template variables. |
|  `template_variables` |  (dict) A dictionary of template variable schemas to be used with the config. Expected format of: `{ "var-name": { "schema": { "type": ("string", "number", or "integer"), "default": (optional value), "minimum": (optional minimum), "maximum": (optional maximum), "enum": [..."(options)"] } } }` |

| Returns |  |
| :--- | :--- |
|  The newly created `RunQueue` |

| Raises |  |
| :--- | :--- |
|  ValueError if any of the parameters are invalid wandb.Error on wandb API errors |

### `create_team`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L874-L884)

```python
create_team(
    team, admin_username=None
)
```

Create a new team.

| Args |  |
| :--- | :--- |
|  `team` |  (str) The name of the team |
|  `admin_username` |  (str) optional username of the admin user of the team, defaults to the current user. |

| Returns |  |
| :--- | :--- |
|  A `Team` object |

### `create_user`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L585-L595)

```python
create_user(
    email, admin=(False)
)
```

Create a new user.

| Args |  |
| :--- | :--- |
|  `email` |  (str) The email address of the user |
|  `admin` |  (bool) Whether this user should be a global instance admin |

| Returns |  |
| :--- | :--- |
|  A `User` object |

### `delete_automation`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L2171-L2203)

```python
delete_automation(
    obj: Union['Automation', str]
) -> Literal[True]
```

Delete an automation.

| Args |  |
| :--- | :--- |
|  `obj` |  The automation to delete, or its ID. |

| Returns |  |
| :--- | :--- |
|  True if the automation was deleted successfully. |

### `flush`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L662-L669)

```python
flush()
```

Flush the local cache.

The api object keeps a local cache of runs, so if the state of the run may
change while executing your script you must clear the local cache with
`api.flush()` to get the latest values associated with the run.

### `from_path`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L671-L725)

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

| Args |  |
| :--- | :--- |
|  `path` |  (str) The path to the project, run, sweep or report |

| Returns |  |
| :--- | :--- |
|  A `Project`, `Run`, `Sweep`, or `BetaReport` instance. |

| Raises |  |
| :--- | :--- |
|  wandb.Error if path is invalid or the object doesn't exist |

### `integrations`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L1671-L1692)

```python
integrations(
    entity: Optional[str] = None,
    *,
    per_page: int = 50
) -> Iterator['Integration']
```

Return an iterator of all integrations for an entity.

| Args |  |
| :--- | :--- |
|  `entity` |  The entity (e.g. team name) for which to fetch integrations. If not provided, the user's default entity will be used. |
|  `per_page` |  Number of integrations to fetch per page. Defaults to 50. Usually there is no reason to change this. |

| Yields |  |
| :--- | :--- |
|  Iterator[SlackIntegration | WebhookIntegration]: An iterator of any supported integrations. |

### `job`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L1356-L1373)

```python
job(
    name: Optional[str],
    path: Optional[str] = None
) -> "public.Job"
```

Return a `Job` from the given parameters.

| Args |  |
| :--- | :--- |
|  `name` |  (str) The job name. |
|  `path` |  (str, optional) If given, the root path in which to download the job artifact. |

| Returns |  |
| :--- | :--- |
|  A `Job` object. |

### `list_jobs`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L1375-L1451)

```python
list_jobs(
    entity: str,
    project: str
) -> List[Dict[str, Any]]
```

Return a list of jobs, if any, for the given entity and project.

| Args |  |
| :--- | :--- |
|  `entity` |  (str) The entity for the listed job(s). |
|  `project` |  (str) The project for the listed job(s). |

| Returns |  |
| :--- | :--- |
|  A list of matching jobs. |

### `project`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L817-L840)

```python
project(
    name: str,
    entity: Optional[str] = None
) -> "public.Project"
```

Return the `Project` with the given name (and entity, if given).

| Args |  |
| :--- | :--- |
|  `name` |  (str) The project name. |
|  `entity` |  (str) Name of the entity requested. If None, will fall back to the default entity passed to `Api`. If no default entity, will raise a `ValueError`. |

| Returns |  |
| :--- | :--- |
|  A `Project` object. |

### `projects`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L792-L815)

```python
projects(
    entity: Optional[str] = None,
    per_page: int = 200
) -> "public.Projects"
```

Get projects for a given entity.

| Args |  |
| :--- | :--- |
|  `entity` |  (str) Name of the entity requested. If None, will fall back to the default entity passed to `Api`. If no default entity, will raise a `ValueError`. |
|  `per_page` |  (int) Sets the page size for query pagination. Usually there is no reason to change this. |

| Returns |  |
| :--- | :--- |
|  A `Projects` object which is an iterable collection of `Project` objects. |

### `queued_run`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L1085-L1106)

```python
queued_run(
    entity, project, queue_name, run_queue_item_id, project_queue=None,
    priority=None
)
```

Return a single queued run based on the path.

Parses paths of the form entity/project/queue_id/run_queue_item_id.

### `registries`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L1496-L1557)

```python
registries(
    organization: Optional[str] = None,
    filter: Optional[Dict[str, Any]] = None
) -> Registries
```

Returns a Registry iterator.

Use the iterator to search and filter registries, collections,
or artifact versions across your organization's registry.

#### Examples:

Find all registries with the names that contain "model"

```python
import wandb

api = wandb.Api()  # specify an org if your entity belongs to multiple orgs
api.registries(filter={"name": {"$regex": "model"}})
```

Find all collections in the registries with the name "my_collection" and the tag "my_tag"

```python
api.registries().collections(filter={"name": "my_collection", "tag": "my_tag"})
```

Find all artifact versions in the registries with a collection name that contains "my_collection" and a version that has the alias "best"

```python
api.registries().collections(
    filter={"name": {"$regex": "my_collection"}}
).versions(filter={"alias": "best"})
```

Find all artifact versions in the registries that contain "model" and have the tag "prod" or alias "best"

```python
api.registries(filter={"name": {"$regex": "model"}}).versions(
    filter={"$or": [{"tag": "prod"}, {"alias": "best"}]}
)
```

| Args |  |
| :--- | :--- |
|  `organization` |  (str, optional) The organization of the registry to fetch. If not specified, use the organization specified in the user's settings. |
|  `filter` |  (dict, optional) MongoDB-style filter to apply to each object in the registry iterator. Fields available to filter for collections are `name`, `description`, `created_at`, `updated_at`. Fields available to filter for collections are `name`, `tag`, `description`, `created_at`, `updated_at` Fields available to filter for versions are `tag`, `alias`, `created_at`, `updated_at`, `metadata` |

| Returns |  |
| :--- | :--- |
|  A registry iterator. |

### `registry`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L1559-L1595)

```python
registry(
    name: str,
    organization: Optional[str] = None
) -> Registry
```

Return a registry given a registry name.

| Args |  |
| :--- | :--- |
|  `name` |  The name of the registry. This is without the `wandb-registry-` prefix. |
|  `organization` |  The organization of the registry. If no organization is set in the settings, the organization will be fetched from the entity if the entity only belongs to one organization. |

| Returns |  |
| :--- | :--- |
|  A registry object. |

#### Examples:

Fetch and update a registry

```python
import wandb

api = wandb.Api()
registry = api.registry(name="my-registry", organization="my-org")
registry.description = "This is an updated description"
registry.save()
```

### `reports`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L842-L872)

```python
reports(
    path: str = "",
    name: Optional[str] = None,
    per_page: int = 50
) -> "public.Reports"
```

Get reports for a given project path.

WARNING: This api is in beta and will likely change in a future release

| Args |  |
| :--- | :--- |
|  `path` |  (str) path to project the report resides in, should be in the form: "entity/project" |
|  `name` |  (str, optional) optional name of the report requested. |
|  `per_page` |  (int) Sets the page size for query pagination. Usually there is no reason to change this. |

| Returns |  |
| :--- | :--- |
|  A `Reports` object which is an iterable collection of `BetaReport` objects. |

### `run`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L1068-L1083)

```python
run(
    path=""
)
```

Return a single run by parsing path in the form entity/project/run_id.

| Args |  |
| :--- | :--- |
|  `path` |  (str) path to run in the form `entity/project/run_id`. If `api.entity` is set, this can be in the form `project/run_id` and if `api.project` is set this can just be the run_id. |

| Returns |  |
| :--- | :--- |
|  A `Run` object. |

### `run_queue`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L1108-L1121)

```python
run_queue(
    entity, name
)
```

Return the named `RunQueue` for entity.

To create a new `RunQueue`, use `wandb.Api().create_run_queue(...)`.

### `runs`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L935-L1066)

```python
runs(
    path: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    order: str = "+created_at",
    per_page: int = 50,
    include_sweeps: bool = (True)
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

Additionally, you can filter by items in the run config or summary metrics.
Such as `config.experiment_name`, `summary_metrics.loss`, etc.

For more complex filtering, you can use MongoDB query operators.
For details, see: https://docs.mongodb.com/manual/reference/operator/query
The following operations are supported:

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

#### Examples:

Find runs in my_project where config.experiment_name has been set to "foo"

```
api.runs(
    path="my_entity/my_project",
    filters={"config.experiment_name": "foo"},
)
```

Find runs in my_project where config.experiment_name has been set to "foo" or "bar"

```
api.runs(
    path="my_entity/my_project",
    filters={
        "$or": [
            {"config.experiment_name": "foo"},
            {"config.experiment_name": "bar"},
        ]
    },
)
```

Find runs in my_project where config.experiment_name matches a regex (anchors are not supported)

```
api.runs(
    path="my_entity/my_project",
    filters={"config.experiment_name": {"$regex": "b.*"}},
)
```

Find runs in my_project where the run name matches a regex (anchors are not supported)

```
api.runs(
    path="my_entity/my_project",
    filters={"display_name": {"$regex": "^foo.*"}},
)
```

Find runs in my_project where config.experiment contains a nested field "category" with value "testing"

```
api.runs(
    path="my_entity/my_project",
    filters={"config.experiment.category": "testing"},
)
```

Find runs in my_project with a loss value of 0.5 nested in a dictionary under model1 in the summary metrics

```
api.runs(
    path="my_entity/my_project",
    filters={"summary_metrics.model1.loss": 0.5},
)
```

Find runs in my_project sorted by ascending loss

```
api.runs(path="my_entity/my_project", order="+summary_metrics.loss")
```

| Args |  |
| :--- | :--- |
|  `path` |  (str) path to project, should be in the form: "entity/project" |
|  `filters` |  (dict) queries for specific runs using the MongoDB query language. You can filter by run properties such as config.key, summary_metrics.key, state, entity, createdAt, etc. For example: `{"config.experiment_name": "foo"}` would find runs with a config entry of experiment name set to "foo" |
|  `order` |  (str) Order can be `created_at`, `heartbeat_at`, `config.*.value`, or `summary_metrics.*`. If you prepend order with a + order is ascending. If you prepend order with a - order is descending (default). The default order is run.created_at from oldest to newest. |
|  `per_page` |  (int) Sets the page size for query pagination. |
|  `include_sweeps` |  (bool) Whether to include the sweep runs in the results. |

| Returns |  |
| :--- | :--- |
|  A `Runs` object, which is an iterable collection of `Run` objects. |

### `slack_integrations`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L1735-L1774)

```python
slack_integrations(
    *,
    entity: Optional[str] = None,
    per_page: int = 50
) -> Iterator['SlackIntegration']
```

Returns an iterator of Slack integrations for an entity.

| Args |  |
| :--- | :--- |
|  `entity` |  The entity (e.g. team name) for which to fetch integrations. If not provided, the user's default entity will be used. |
|  `per_page` |  Number of integrations to fetch per page. Defaults to 50. Usually there is no reason to change this. |

| Yields |  |
| :--- | :--- |
|  Iterator[SlackIntegration]: An iterator of Slack integrations. |

#### Examples:

Get all registered Slack integrations for the team "my-team":

```python
import wandb

api = wandb.Api()
slack_integrations = api.slack_integrations(entity="my-team")
```

Find only Slack integrations that post to channel names starting with "team-alerts-":

```python
slack_integrations = api.slack_integrations(entity="my-team")
team_alert_integrations = [
    ig
    for ig in slack_integrations
    if ig.channel_name.startswith("team-alerts-")
]
```

### `sweep`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L1123-L1138)

```python
sweep(
    path=""
)
```

Return a sweep by parsing path in the form `entity/project/sweep_id`.

| Args |  |
| :--- | :--- |
|  `path` |  (str, optional) path to sweep in the form entity/project/sweep_id. If `api.entity` is set, this can be in the form project/sweep_id and if `api.project` is set this can just be the sweep_id. |

| Returns |  |
| :--- | :--- |
|  A `Sweep` object. |

### `sync_tensorboard`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L597-L619)

```python
sync_tensorboard(
    root_dir, run_id=None, project=None, entity=None
)
```

Sync a local directory containing tfevent files to wandb.

### `team`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L886-L895)

```python
team(
    team: str
) -> "public.Team"
```

Return the matching `Team` with the given name.

| Args |  |
| :--- | :--- |
|  `team` |  (str) The name of the team. |

| Returns |  |
| :--- | :--- |
|  A `Team` object. |

### `update_automation`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L2049-L2169)

```python
update_automation(
    obj: "Automation",
    *,
    create_missing: bool = (False),
    **kwargs
) -> "Automation"
```

Update an existing automation.

| Args |  |
| :--- | :--- |
|  `obj` |  The automation to update. Must be an existing automation. create_missing (bool): If True, and the automation does not exist, create it. |
|  `**kwargs` |  Any additional values to assign to the automation before updating it. If given, these will override any values that may already be set on the automation: - `name`: The name of the automation. - `description`: The description of the automation. - `enabled`: Whether the automation is enabled. - `scope`: The scope of the automation. - `event`: The event that triggers the automation. - `action`: The action that is triggered by the automation. |

| Returns |  |
| :--- | :--- |
|  The updated automation. |

#### Examples:

Disable and edit the description of an existing automation ("my-automation"):

```python
import wandb

api = wandb.Api()

automation = api.automation(name="my-automation")
automation.enabled = False
automation.description = "Kept for reference, but no longer used."

updated_automation = api.update_automation(automation)
```

* <b>`OR`</b>: ```python
  import wandb

api = wandb.Api()

automation = api.automation(name="my-automation")

updated_automation = api.update_automation(
automation,
enabled=False,
description="Kept for reference, but no longer used.",
)

```


### `upsert_run_queue`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L470-L583)

```python
upsert_run_queue(
    name: str,
    resource_config: dict,
    resource_type: "public.RunQueueResourceType",
    entity: Optional[str] = None,
    template_variables: Optional[dict] = None,
    external_links: Optional[dict] = None,
    prioritization_mode: Optional['public.RunQueuePrioritizationMode'] = None
)
```

Upsert a run queue (launch).

| Args |  |
| :--- | :--- |
|  `name` |  (str) Name of the queue to create |
|  `entity` |  (str) Optional name of the entity to create the queue. If None, will use the configured or default entity. |
|  `resource_config` |  (dict) Optional default resource configuration to be used for the queue. Use handlebars (eg. `{{var}}`) to specify template variables. |
|  `resource_type` |  (str) Type of resource to be used for the queue. One of "local-container", "local-process", "kubernetes", "sagemaker", or "gcp-vertex". |
|  `template_variables` |  (dict) A dictionary of template variable schemas to be used with the config. Expected format of: `{ "var-name": { "schema": { "type": ("string", "number", or "integer"), "default": (optional value), "minimum": (optional minimum), "maximum": (optional maximum), "enum": [..."(options)"] } } }` |
|  `external_links` |  (dict) Optional dictionary of external links to be used with the queue. Expected format of: `{ "name": "url" }` |
|  `prioritization_mode` |  (str) Optional version of prioritization to use. Either "V0" or None |

| Returns |  |
| :--- | :--- |
|  The upserted `RunQueue`. |

| Raises |  |
| :--- | :--- |
|  ValueError if any of the parameters are invalid wandb.Error on wandb API errors |

### `user`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L897-L917)

```python
user(
    username_or_email: str
) -> Optional['public.User']
```

Return a user from a username or email address.

Note: This function only works for Local Admins, if you are trying to get your own user object, please use `api.viewer`.

| Args |  |
| :--- | :--- |
|  `username_or_email` |  (str) The username or email address of the user |

| Returns |  |
| :--- | :--- |
|  A `User` object or None if a user couldn't be found |

### `users`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L919-L933)

```python
users(
    username_or_email: str
) -> List['public.User']
```

Return all users from a partial username or email address query.

Note: This function only works for Local Admins, if you are trying to get your own user object, please use `api.viewer`.

| Args |  |
| :--- | :--- |
|  `username_or_email` |  (str) The prefix or suffix of the user you want to find |

| Returns |  |
| :--- | :--- |
|  An array of `User` objects |

### `webhook_integrations`

[View source](https://www.github.com/wandb/wandb/tree/v0.20.1/wandb/apis/public/api.py#L1694-L1733)

```python
webhook_integrations(
    entity: Optional[str] = None,
    *,
    per_page: int = 50
) -> Iterator['WebhookIntegration']
```

Returns an iterator of webhook integrations for an entity.

| Args |  |
| :--- | :--- |
|  `entity` |  The entity (e.g. team name) for which to fetch integrations. If not provided, the user's default entity will be used. |
|  `per_page` |  Number of integrations to fetch per page. Defaults to 50. Usually there is no reason to change this. |

| Yields |  |
| :--- | :--- |
|  Iterator[WebhookIntegration]: An iterator of webhook integrations. |

#### Examples:

Get all registered webhook integrations for the team "my-team":

```python
import wandb

api = wandb.Api()
webhook_integrations = api.webhook_integrations(entity="my-team")
```

Find only webhook integrations that post requests to "https://my-fake-url.com":

```python
webhook_integrations = api.webhook_integrations(entity="my-team")
my_webhooks = [
    ig
    for ig in webhook_integrations
    if ig.url_endpoint.startswith("https://my-fake-url.com")
]
```

| Class Variables |  |
| :--- | :--- |
|  `CREATE_PROJECT`<a id="CREATE_PROJECT"></a> |   |
|  `DEFAULT_ENTITY_QUERY`<a id="DEFAULT_ENTITY_QUERY"></a> |   |
|  `USERS_QUERY`<a id="USERS_QUERY"></a> |   |
|  `VIEWER_QUERY`<a id="VIEWER_QUERY"></a> |   |
