---
menu:
  default:
    identifier: registry_cards
    parent: registry
title: Annotate collections
weight: 8
---

Add human-friendly text to your collections to help users understand the purpose of the collection and the artifacts it contains. 


Depending on the collection, you might want to include information about the training data, model architecture, task, license, references, and deployment. The proceeding lists some topics worth documenting in a collection:



W&B recommends including at minimum these details:
* **Summary**: The purpose of the collection. The machine learning framework used for the machine learning experiment.
* **License**: The legal terms and permissions associated with the use of the machine learning model. It helps model users understand the legal framework under which they can utilize the model. Common licenses include Apache 2.0, MIT, and GPL.
* **References**: Citations or references to relevant research papers, datasets, or external resources.

If your collection contains training data, consider including these additional details:
* **Training data**: Describe the training data used
* **Processing**: Processing done on the training data set.
* **Data storage**: Where is that data stored and how to access it.


If your collection contains a machine learning model, consider including these additional details:
* **Architecture**: Information about the model architecture, layers, and any specific design choices.
* **Task**: The specific type of task or problem that the machine that the collection model is designed to perform. It's a categorization of the model's intended capability.
* **Deserialize the model**: Provide information on how someone on your team can load the model into memory.
* **Task**: The specific type of task or problem that the machine learning model is designed to perform. It's a categorization of the model's intended capability.
* **Deployment**: Details on how and where the model is deployed and guidance on how the model is integrated into other enterprise systems, such as a workflow orchestration platforms.


## Add a description to a collection

Interactively or programmatically add a description to a collection with the W&B Registry UI or Python SDK.

{{< tabpane text=true >}}
  {{% tab header="W&B Registry UI" %}}
1. Navigate to W&B Registry at [https://wandb.ai/registry/](https://wandb.ai/registry/).
2. Click on a collection.
3. Select **View details** next to the name of the collection.
4. Within the **Description** field, provide information about your collection. Format text within with [Markdown markup language](https://www.markdownguide.org/).

  {{% /tab %}}
  {{% tab header="Python SDK" %}}

Use the [`wandb.Api().artifact_collection()`]({{< relref "/ref/python/public-api/api.md#artifact_collection" >}}) method to access a collection's description. Use the returned object's `description` property to add, or update, a description to the collection.

Specify the collection's type for the `type_name` parameter and the collection's full name for the `name` parameter. A collection's name consists of the prefix “wandb-registry”, the name of the registry, and the name of the collection separated by a forward slashes:

```text
wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}
```

Copy and paste the proceeding code snippet into your Python script or notebook. Replace values enclosed in angle brackets (`<>`) with your own.

```python
import wandb

api = wandb.Api()

collection = api.artifact_collection(
  type_name = "<collection_type>", 
  name = "<collection_name>"
  )


collection.description = "This is a description."
collection.save()  
```  
  {{% /tab %}}
{{< /tabpane >}}

For example, the proceeding image shows a collection that documents a model's architecture, intended use, performance information and more.

{{< img src="/images/registry/registry_card.png" alt="Collection card with information about the model architecture, intended use, performance information and more." >}}  