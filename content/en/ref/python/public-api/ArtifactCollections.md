---
title: ArtifactCollections
namespace: public_apis_namespace
python_object_type: class
---
{{< readfile file="/_includes/public-api-use.md" >}}


{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/artifacts.py >}}




## <kbd>class</kbd> `ArtifactCollections`
Artifact collections of a specific type in a project. 



**Args:**
 
 - `client`:  The client instance to use for querying W&B. 
 - `entity`:  The entity (user or team) that owns the project. 
 - `project`:  The name of the project to query for artifact collections. 
 - `type_name`:  The name of the artifact type for which to fetch collections. 
 - `per_page`:  The number of artifact collections to fetch per page. Default is 50. 


### <kbd>property</kbd> ArtifactCollections.length





---



