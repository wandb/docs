---
menu:
  default:
    identifier: registry_cards
    parent: registry
title: Document collections
weight: 9
---

Add human-friendly text to your collections to help users understand the purpose of the collection and the artifacts it contains. 


Depending on the collection, you might want to include information about the training data, model architecture, task, license, references, and deployment. The proceeding lists some topics worth documenting in a collection:



In general, a collection description should include:
* **Summary**: The purpose of the collection. The machine learning framework used for the machine learning experiment.
* **License**: The legal terms and permissions associated with the use of the machine learning model. It helps model users understand the legal framework under which they can utilize the model.
* **References**: Citations or references to relevant research papers, datasets, or external resources.

If your collections training data, you might want to include:
* **Training data**: Describe the training data used
* **Processing**: Processing done on the training data set.
* **Data storage**: Where is that data stored and how to access it.


If your collection contains a machine learning model, you might want to include:
* **Architecture**: Information about the model architecture, layers, and any specific design choices.
* **Task**: The specific type of task or problem that the machine that the collection model is designed to perform. It's a categorization of the model's intended capability.
* **Deserialize the model**: Provide information on how someone on your team can load the model into memory.
* **Task**: The specific type of task or problem that the machine learning model is designed to perform. It's a categorization of the model's intended capability.
* **Deployment**: Details on how and where the model is deployed and guidance on how the model is integrated into other enterprise systems, such as a workflow orchestration platforms.


## Add a description to a collection

1. Navigate to W&B Registry at [https://wandb.ai/registry/](https://wandb.ai/registry/).
2. Click on a collection.
3. Select **View details** next to the name of the collection.
4. Within the **Description** field, provide information about your collection. Format text within with [Markdown markup language](https://www.markdownguide.org/).

For example, the following images shows the model card of a **Credit-card Default Prediction** registered model.
{{< img src="/images/models/model_card_credit_example.png" alt="" >}}