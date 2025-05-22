---
title: projects
object_type: public_apis_namespace
data_type_classification: module
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/apis/public/projects.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API for Project objects. 

This module provides classes for interacting with W&B projects and their associated data. 



**Example:**
 ```python
from wandb.apis.public import Api

# Initialize API
api = Api()

# Get all projects for an entity
projects = api.projects("entity")

# Access project data
for project in projects:
     print(f"Project: {project.name}")
     print(f"URL: {project.url}")

     # Get artifact types
     for artifact_type in project.artifacts_types():
         print(f"Artifact Type: {artifact_type.name}")

     # Get sweeps
     for sweep in project.sweeps():
         print(f"Sweep ID: {sweep.id}")
         print(f"State: {sweep.state}")
``` 



**Note:**

> This module is part of the W&B Public API and provides methods to access and manage projects. For creating new projects, use wandb.init() with a new project name. 

