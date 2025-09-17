---
title: Artifacts
namespace: public_apis_namespace
python_object_type: class
---
{{< readfile file="/_includes/public-api-use.md" >}}


{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/artifacts.py >}}




## <kbd>class</kbd> `Artifacts`
An iterable collection of artifact versions associated with a project. 

Optionally pass in filters to narrow down the results based on specific criteria. 



**Args:**
 
 - `client`:  The client instance to use for querying W&B. 
 - `entity`:  The entity (user or team) that owns the project. 
 - `project`:  The name of the project to query for artifacts. 
 - `collection_name`:  The name of the artifact collection to query. 
 - `type`:  The type of the artifacts to query. Common examples include  "dataset" or "model". 
 - `filters`:  Optional mapping of filters to apply to the query. 
 - `order`:  Optional string to specify the order of the results. 
 - `per_page`:  The number of artifact versions to fetch per page. Default is 50. 
 - `tags`:  Optional string or list of strings to filter artifacts by tags. 


### <kbd>property</kbd> Artifacts.length





---


