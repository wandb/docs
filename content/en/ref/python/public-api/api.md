---
title: Api
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L151-L2513 >}}

Used for querying the W&B server.

#### Examples:

```python
import wandb

wandb.Api()
```

| Attributes |  |
| :--- | :--- |
|  `client` |  Returns the client object. |
|  `default_entity` |  Returns the default W&B entity. |
|  `user_agent` |  Returns W&B public user agent. |
|  `viewer` |  Returns the viewer object. |

## Methods

### `artifact`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L1568-L1615)

```python
artifact(
    name: str,
    type: (str | None) = None
)
```

Returns a single artifact.

| Args |  |
| :--- | :--- |
|  `name` |  The artifact's name. The name of an artifact resembles a filepath that consists, at a minimum, the name of the project the artifact was logged to, the name of the artifact, and the artifact's version or alias. Optionally append the entity that logged the artifact as a prefix followed by a forward slash. If no entity is specified in the name, the Run or API setting's entity is used. |
|  `type` |  The type of artifact to fetch. |

| Returns |  |
| :--- | :--- |
|  An `Artifact` object. |

| Raises |  |
| :--- | :--- |
|  `ValueError` |  If the artifact name is not specified. |
|  `ValueError` |  If the artifact type is specified but does not match the type of the fetched artifact. |

#### Examples:

In the proceeding code snippets "entity", "project", "artifact",
"version", and "alias" are placeholders for your W&B entity, name
of the project the artifact is in, the name of the artifact,
and artifact's version, respectively.

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

#### Note:

This method is intended for external use only. Do not call `api.artifact()` within the wandb repository code.

### `artifact_collection`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L1403-L1457)

```python
artifact_collection(
    type_name: str,
    name: str
) -> ArtifactCollection
```

Returns a single artifact collection by type.

You can use the returned `ArtifactCollection` object to retrieve
information about specific artifacts in that collection, and more.

| Args |  |
| :--- | :--- |
|  `type_name` |  The type of artifact collection to fetch. |
|  `name` |  An artifact collection name. Optionally append the entity that logged the artifact as a prefix followed by a forward slash. |

| Returns |  |
| :--- | :--- |
|  An `ArtifactCollection` object. |

#### Examples:

In the proceeding code snippet "type", "entity", "project", and
"artifact_name" are placeholders for the collection type, your W&B
entity, name of the project the artifact is in, and the name of
the artifact, respectively.

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

### `artifact_collection_exists`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L1752-L1784)

```python
artifact_collection_exists(
    name: str,
    type: str
) -> bool
```

Whether an artifact collection exists within a specified project and entity.

| Args |  |
| :--- | :--- |
|  `name` |  An artifact collection name. Optionally append the entity that logged the artifact as a prefix followed by a forward slash. If entity or project is not specified, infer the collection from the override params if they exist. Otherwise, entity is pulled from the user settings and project will default to "uncategorized". |
|  `type` |  The type of artifact collection. |

| Returns |  |
| :--- | :--- |
|  True if the artifact collection exists, False otherwise. |

#### Examples:

In the proceeding code snippet "type", and "collection_name" refer to the type
of the artifact collection and the name of the collection, respectively.

```python
import wandb

wandb.Api.artifact_collection_exists(type="type", name="collection_name")
```

### `artifact_collections`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L1374-L1401)

```python
artifact_collections(
    project_name: str,
    type_name: str,
    per_page: int = 50
) -> ArtifactCollections
```

Returns a collection of matching artifact collections.

| Args |  |
| :--- | :--- |
|  `project_name` |  The name of the project to filter on. |
|  `type_name` |  The name of the artifact type to filter on. |
|  `per_page` |  Sets the page size for query pagination. None will use the default size. Usually there is no reason to change this. |

| Returns |  |
| :--- | :--- |
|  An iterable `ArtifactCollections` object. |

### `artifact_exists`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L1714-L1750)

```python
artifact_exists(
    name: str,
    type: (str | None) = None
) -> bool
```

Whether an artifact version exists within the specified project and entity.

| Args |  |
| :--- | :--- |
|  `name` |  The name of artifact. Add the artifact's entity and project as a prefix. Append the version or the alias of the artifact with a colon. If the entity or project is not specified, W&B uses override parameters if populated. Otherwise, the entity is pulled from the user settings and the project is set to "Uncategorized". |
|  `type` |  The type of artifact. |

| Returns |  |
| :--- | :--- |
|  True if the artifact version exists, False otherwise. |

#### Examples:

In the proceeding code snippets "entity", "project", "artifact",
"version", and "alias" are placeholders for your W&B entity, name of
the project the artifact is in, the name of the artifact, and
artifact's version, respectively.

```python
import wandb

wandb.Api().artifact_exists("entity/project/artifact:version")
wandb.Api().artifact_exists("entity/project/artifact:alias")
```

### `artifact_type`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L1350-L1372)

```python
artifact_type(
    type_name: str,
    project: (str | None) = None
) -> ArtifactType
```

Returns the matching `ArtifactType`.

| Args |  |
| :--- | :--- |
|  `type_name` |  The name of the artifact type to retrieve. |
|  `project` |  If given, a project name or path to filter on. |

| Returns |  |
| :--- | :--- |
|  An `ArtifactType` object. |

### `artifact_types`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L1327-L1348)

```python
artifact_types(
    project: (str | None) = None
) -> ArtifactTypes
```

Returns a collection of matching artifact types.

| Args |  |
| :--- | :--- |
|  `project` |  The project name or path to filter on. |

| Returns |  |
| :--- | :--- |
|  An iterable `ArtifactTypes` object. |

### `artifact_versions`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L1459-L1469)

```python
artifact_versions(
    type_name, name, per_page=50
)
```

Deprecated. Use `Api.artifacts(type_name, name)` method instead.

### `artifacts`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L1471-L1524)

```python
artifacts(
    type_name: str,
    name: str,
    per_page: int = 50,
    tags: (list[str] | None) = None
) -> Artifacts
```

Return an `Artifacts` collection.

| Args |  |
| :--- | :--- |

type_name: The type of artifacts to fetch.
name: The artifact's collection name. Optionally append the
entity that logged the artifact as a prefix followed by
a forward slash.
per_page: Sets the page size for query pagination. If set to
`None`, use the default size. Usually there is no reason
to change this.
tags: Only return artifacts with all of these tags.

| Returns |  |
| :--- | :--- |
|  An iterable `Artifacts` object. |

#### Examples:

In the proceeding code snippet, "type", "entity", "project", and
"artifact_name" are placeholders for the artifact type, W&B entity,
name of the project the artifact was logged to,
and the name of the artifact, respectively.

```python
import wandb

wandb.Api().artifacts(type_name="type", name="entity/project/artifact_name")
```

### `automation`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L2154-L2190)

```python
automation(
    name: str,
    *,
    entity: (str | None) = None
) -> Automation
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

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L2192-L2247)

```python
automations(
    entity: (str | None) = None,
    *,
    name: (str | None) = None,
    per_page: int = 50
) -> Iterator[Automation]
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

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L2249-L2355)

```python
create_automation(
    obj: NewAutomation,
    *,
    fetch_existing: bool = (False),
    **kwargs
) -> Automation
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

### `create_custom_chart`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L556-L633)

```python
create_custom_chart(
    entity: str,
    name: str,
    display_name: str,
    spec_type: Literal['vega2'],
    access: Literal['private', 'public'],
    spec: (str | dict)
) -> str
```

Create a custom chart preset and return its id.

| Args |  |
| :--- | :--- |
|  `entity` |  The entity (user or team) that owns the chart |
|  `name` |  Unique identifier for the chart preset |
|  `display_name` |  Human-readable name shown in the UI |
|  `spec_type` |  Type of specification. Must be "vega2" for Vega-Lite v2 specifications. |
|  `access` |  Access level for the chart: - "private": Chart is only accessible to the entity that created it - "public": Chart is publicly accessible |
|  `spec` |  The Vega/Vega-Lite specification as a dictionary or JSON string |

| Returns |  |
| :--- | :--- |
|  The ID of the created chart preset in the format "entity/name" |

| Raises |  |
| :--- | :--- |
|  `wandb.Error` |  If chart creation fails |
|  `UnsupportedError` |  If the server doesn't support custom charts |

#### Example:

```python
import wandb

api = wandb.Api()

# Define a simple bar chart specification
vega_spec = {
    "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
    "mark": "bar",
    "data": {"name": "wandb"},
    "encoding": {
        "x": {"field": "${field:x}", "type": "ordinal"},
        "y": {"field": "${field:y}", "type": "quantitative"},
    },
}

# Create the custom chart
chart_id = api.create_custom_chart(
    entity="my-team",
    name="my-bar-chart",
    display_name="My Custom Bar Chart",
    spec_type="vega2",
    access="private",
    spec=vega_spec,
)

# Use with wandb.plot_table()
chart = wandb.plot_table(
    vega_spec_name=chart_id,
    data_table=my_table,
    fields={"x": "category", "y": "value"},
)
```

### `create_project`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L416-L423)

```python
create_project(
    name: str,
    entity: str
) -> None
```

Create a new project.

| Args |  |
| :--- | :--- |
|  `name` |  The name of the new project. |
|  `entity` |  The entity of the new project. |

### `create_registry`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L1894-L1967)

```python
create_registry(
    name: str,
    visibility: Literal['organization', 'restricted'],
    organization: (str | None) = None,
    description: (str | None) = None,
    artifact_types: (list[str] | None) = None
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

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L425-L447)

```python
create_run(
    *,
    run_id: (str | None) = None,
    project: (str | None) = None,
    entity: (str | None) = None
) -> public.Run
```

Create a new run.

| Args |  |
| :--- | :--- |
|  `run_id` |  The ID to assign to the run. If not specified, W&B creates a random ID. |
|  `project` |  The project where to log the run to. If no project is specified, log the run to a project called "Uncategorized". |
|  `entity` |  The entity that owns the project. If no entity is specified, log the run to the default entity. |

| Returns |  |
| :--- | :--- |
|  The newly created `Run`. |

### `create_run_queue`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L449-L554)

```python
create_run_queue(
    name: str,
    type: public.RunQueueResourceType,
    entity: (str | None) = None,
    prioritization_mode: (public.RunQueuePrioritizationMode | None) = None,
    config: (dict | None) = None,
    template_variables: (dict | None) = None
) -> public.RunQueue
```

Create a new run queue in W&B Launch.

| Args |  |
| :--- | :--- |
|  `name` |  Name of the queue to create |
|  `type` |  Type of resource to be used for the queue. One of "local-container", "local-process", "kubernetes","sagemaker", or "gcp-vertex". |
|  `entity` |  Name of the entity to create the queue. If `None`, use the configured or default entity. |
|  `prioritization_mode` |  Version of prioritization to use. Either "V0" or `None`. |
|  `config` |  Default resource configuration to be used for the queue. Use handlebars (eg. `{{var}}`) to specify template variables. |
|  `template_variables` |  A dictionary of template variable schemas to use with the config. |

| Returns |  |
| :--- | :--- |
|  The newly created `RunQueue`. |

| Raises |  |
| :--- | :--- |
|  `ValueError` if any of the parameters are invalid `wandb.Error` on wandb API errors |

### `create_team`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L1050-L1063)

```python
create_team(
    team: str,
    admin_username: (str | None) = None
) -> Team
```

Create a new team.

| Args |  |
| :--- | :--- |
|  `team` |  The name of the team |
|  `admin_username` |  Username of the admin user of the team. Defaults to the current user. |

| Returns |  |
| :--- | :--- |
|  A `Team` object. |

### `create_user`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L744-L756)

```python
create_user(
    email: str,
    admin: (bool | None) = (False)
) -> User
```

Create a new user.

| Args |  |
| :--- | :--- |
|  `email` |  The email address of the user. |
|  `admin` |  Set user as a global instance administrator. |

| Returns |  |
| :--- | :--- |
|  A `User` object. |

### `delete_automation`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L2480-L2513)

```python
delete_automation(
    obj: (Automation | str)
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

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L823-L830)

```python
flush()
```

Flush the local cache.

The api object keeps a local cache of runs, so if the state of the run
may change while executing your script you must clear the local cache
with `api.flush()` to get the latest values associated with the run.

### `from_path`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L832-L894)

```python
from_path(
    path: str
)
```

Return a run, sweep, project or report from a path.

| Args |  |
| :--- | :--- |
|  `path` |  The path to the project, run, sweep or report |

| Returns |  |
| :--- | :--- |
|  A `Project`, `Run`, `Sweep`, or `BetaReport` instance. |

| Raises |  |
| :--- | :--- |
|  `wandb.Error` if path is invalid or the object doesn't exist. |

#### Examples:

In the proceeding code snippets "project", "team", "run_id", "sweep_id",
and "report_name" are placeholders for the project, team, run ID,
sweep ID, and the name of a specific report, respectively.

```python
import wandb

api = wandb.Api()

project = api.from_path("project")
team_project = api.from_path("team/project")
run = api.from_path("team/project/runs/run_id")
sweep = api.from_path("team/project/sweeps/sweep_id")
report = api.from_path("team/project/reports/report_name")
```

### `integrations`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L1969-L1991)

```python
integrations(
    entity: (str | None) = None,
    *,
    per_page: int = 50
) -> Iterator[Integration]
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

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L1617-L1634)

```python
job(
    name: (str | None),
    path: (str | None) = None
) -> public.Job
```

Return a `Job` object.

| Args |  |
| :--- | :--- |
|  `name` |  The name of the job. |
|  `path` |  The root path to download the job artifact. |

| Returns |  |
| :--- | :--- |
|  A `Job` object. |

### `list_jobs`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L1636-L1712)

```python
list_jobs(
    entity: str,
    project: str
) -> list[dict[str, Any]]
```

Return a list of jobs, if any, for the given entity and project.

| Args |  |
| :--- | :--- |
|  `entity` |  The entity for the listed jobs. |
|  `project` |  The project for the listed jobs. |

| Returns |  |
| :--- | :--- |
|  A list of matching jobs. |

### `project`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L979-L1003)

```python
project(
    name: str,
    entity: (str | None) = None
) -> public.Project
```

Return the `Project` with the given name (and entity, if given).

| Args |  |
| :--- | :--- |
|  `name` |  The project name. |
|  `entity` |  Name of the entity requested. If None, will fall back to the default entity passed to `Api`. If no default entity, will raise a `ValueError`. |

| Returns |  |
| :--- | :--- |
|  A `Project` object. |

### `projects`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L952-L977)

```python
projects(
    entity: (str | None) = None,
    per_page: int = 200
) -> public.Projects
```

Get projects for a given entity.

| Args |  |
| :--- | :--- |
|  `entity` |  Name of the entity requested. If None, will fall back to the default entity passed to `Api`. If no default entity, will raise a `ValueError`. |
|  `per_page` |  Sets the page size for query pagination. If set to `None`, use the default size. Usually there is no reason to change this. |

| Returns |  |
| :--- | :--- |
|  A `Projects` object which is an iterable collection of `Project`objects. |

### `queued_run`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L1271-L1292)

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

### `registries`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L1786-L1852)

```python
registries(
    organization: (str | None) = None,
    filter: (dict[str, Any] | None) = None
) -> Registries
```

Returns a lazy iterator of `Registry` objects.

Use the iterator to search and filter registries, collections,
or artifact versions across your organization's registry.

| Args |  |
| :--- | :--- |
|  `organization` |  (str, optional) The organization of the registry to fetch. If not specified, use the organization specified in the user's settings. |
|  `filter` |  (dict, optional) MongoDB-style filter to apply to each object in the lazy registry iterator. Fields available to filter for registries are `name`, `description`, `created_at`, `updated_at`. Fields available to filter for collections are `name`, `tag`, `description`, `created_at`, `updated_at` Fields available to filter for versions are `tag`, `alias`, `created_at`, `updated_at`, `metadata` |

| Returns |  |
| :--- | :--- |
|  A lazy iterator of `Registry` objects. |

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

### `registry`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L1854-L1892)

```python
registry(
    name: str,
    organization: (str | None) = None
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

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L1005-L1048)

```python
reports(
    path: str = "",
    name: (str | None) = None,
    per_page: int = 50
) -> public.Reports
```

Get reports for a given project path.

Note: `wandb.Api.reports()` API is in beta and will likely change in
future releases.

| Args |  |
| :--- | :--- |
|  `path` |  The path to the project the report resides in. Specify the entity that created the project as a prefix followed by a forward slash. |
|  `name` |  Name of the report requested. |
|  `per_page` |  Sets the page size for query pagination. If set to `None`, use the default size. Usually there is no reason to change this. |

| Returns |  |
| :--- | :--- |
|  A `Reports` object which is an iterable collection of `BetaReport` objects. |

#### Examples:

```python
import wandb

wandb.Api.reports("entity/project")
```

### `run`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L1251-L1269)

```python
run(
    path=""
)
```

Return a single run by parsing path in the form `entity/project/run_id`.

| Args |  |
| :--- | :--- |
|  `path` |  Path to run in the form `entity/project/run_id`. If `api.entity` is set, this can be in the form `project/run_id` and if `api.project` is set this can just be the run_id. |

| Returns |  |
| :--- | :--- |
|  A `Run` object. |

### `run_queue`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L1294-L1307)

```python
run_queue(
    entity: str,
    name: str
)
```

Return the named `RunQueue` for entity.

See `Api.create_run_queue` for more information on how to create a run queue.

### `runs`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L1120-L1249)

```python
runs(
    path: (str | None) = None,
    filters: (dict[str, Any] | None) = None,
    order: str = "+created_at",
    per_page: int = 50,
    include_sweeps: bool = (True),
    lazy: bool = (True)
)
```

Returns a `Runs` object, which lazily iterates over `Run` objects.

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

| Args |  |
| :--- | :--- |
|  `path` |  (str) path to project, should be in the form: "entity/project" |
|  `filters` |  (dict) queries for specific runs using the MongoDB query language. You can filter by run properties such as config.key, summary_metrics.key, state, entity, createdAt, etc. For example: `{"config.experiment_name": "foo"}` would find runs with a config entry of experiment name set to "foo" |
|  `order` |  (str) Order can be `created_at`, `heartbeat_at`, `config.*.value`, or `summary_metrics.*`. If you prepend order with a + order is ascending (default). If you prepend order with a - order is descending. The default order is run.created_at from oldest to newest. |
|  `per_page` |  (int) Sets the page size for query pagination. |
|  `include_sweeps` |  (bool) Whether to include the sweep runs in the results. |
|  `lazy` |  (bool) Whether to use lazy loading for faster performance. When True (default), only essential run metadata is loaded initially. Heavy fields like config, summaryMetrics, and systemMetrics are loaded on-demand when accessed. Set to False for full data upfront. |

| Returns |  |
| :--- | :--- |
|  A `Runs` object, which is an iterable collection of `Run` objects. |

#### Examples:

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

### `slack_integrations`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L2037-L2079)

```python
slack_integrations(
    *,
    entity: (str | None) = None,
    per_page: int = 50
) -> Iterator[SlackIntegration]
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

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L1309-L1325)

```python
sweep(
    path=""
)
```

Return a sweep by parsing path in the form `entity/project/sweep_id`.

| Args |  |
| :--- | :--- |
|  `path` |  Path to sweep in the form entity/project/sweep_id. If `api.entity` is set, this can be in the form project/sweep_id and if `api.project` is set this can just be the sweep_id. |

| Returns |  |
| :--- | :--- |
|  A `Sweep` object. |

### `sync_tensorboard`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L758-L780)

```python
sync_tensorboard(
    root_dir, run_id=None, project=None, entity=None
)
```

Sync a local directory containing tfevent files to wandb.

### `team`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L1065-L1076)

```python
team(
    team: str
) -> Team
```

Return the matching `Team` with the given name.

| Args |  |
| :--- | :--- |
|  `team` |  The name of the team. |

| Returns |  |
| :--- | :--- |
|  A `Team` object. |

### `update_automation`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L2357-L2478)

```python
update_automation(
    obj: Automation,
    *,
    create_missing: bool = (False),
    **kwargs
) -> Automation
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

OR

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

### `upsert_run_queue`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L635-L742)

```python
upsert_run_queue(
    name: str,
    resource_config: dict,
    resource_type: public.RunQueueResourceType,
    entity: (str | None) = None,
    template_variables: (dict | None) = None,
    external_links: (dict | None) = None,
    prioritization_mode: (public.RunQueuePrioritizationMode | None) = None
)
```

Upsert a run queue in W&B Launch.

| Args |  |
| :--- | :--- |
|  `name` |  Name of the queue to create |
|  `entity` |  Optional name of the entity to create the queue. If `None`, use the configured or default entity. |
|  `resource_config` |  Optional default resource configuration to be used for the queue. Use handlebars (eg. `{{var}}`) to specify template variables. |
|  `resource_type` |  Type of resource to be used for the queue. One of "local-container", "local-process", "kubernetes", "sagemaker", or "gcp-vertex". |
|  `template_variables` |  A dictionary of template variable schemas to be used with the config. |
|  `external_links` |  Optional dictionary of external links to be used with the queue. |
|  `prioritization_mode` |  Optional version of prioritization to use. Either "V0" or None |

| Returns |  |
| :--- | :--- |
|  The upserted `RunQueue`. |

| Raises |  |
| :--- | :--- |
|  ValueError if any of the parameters are invalid wandb.Error on wandb API errors |

### `user`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L1078-L1101)

```python
user(
    username_or_email: str
) -> (User | None)
```

Return a user from a username or email address.

This function only works for local administrators. Use `api.viewer`
to get your own user object.

| Args |  |
| :--- | :--- |
|  `username_or_email` |  The username or email address of the user. |

| Returns |  |
| :--- | :--- |
|  A `User` object or None if a user is not found. |

### `users`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L1103-L1118)

```python
users(
    username_or_email: str
) -> list[User]
```

Return all users from a partial username or email address query.

This function only works for local administrators. Use `api.viewer`
to get your own user object.

| Args |  |
| :--- | :--- |
|  `username_or_email` |  The prefix or suffix of the user you want to find. |

| Returns |  |
| :--- | :--- |
|  An array of `User` objects. |

### `webhook_integrations`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.1/wandb/apis/public/api.py#L1993-L2035)

```python
webhook_integrations(
    entity: (str | None) = None,
    *,
    per_page: int = 50
) -> Iterator[WebhookIntegration]
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
