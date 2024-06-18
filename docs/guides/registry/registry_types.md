---
displayed_sidebar: default
---

# Registry types

W&B supports two types of registries: [Core registries](#core-registries) and [Custom registries](#custom-registries). 

## Core registries
A core registry is a template for specific use cases with a standardizes setup to maintain core functionality. W&B automatically creates two core registries: Models and Datasets.

By default, the Models registry is configured to accept model artifact types (`type="model"`) and the Dataset registry is configured to accept dataset artifact types of (`type="dataset"`). An admin can add additional accepted artifact types. 


## Custom registries
Use custom registries for greater flexibility and customization. [INSERT]

For information on how to create a custom registry, see [Create a custom registry](./create_collection.md).


## Summary
The proceeding table summarizes the differences between core and custom registries:

|                | Core  | Custom|
| -------------- | ----- | ----- |
| Visibility     | Organizational visibility only. Visibility can not be altered. | Either organization or restricted. Visibility can be altered from organization to restricted visibility.|
| Metadata       | Preconfigured and not editable by users. | Users can edit.  |
| Artifact types | Preconfigured and accepted artifact types cannot be removed. Users can add additional accepted artifact types. | Admin can define accepted types. |
| Customization    | Can add additional types to the existing list.|  Edit registry name, description, visibility, and accepted artifact types.|

<!-- ## Core registries
Core registries are integrated into the registries platform automatically. There are two types of core registries: models and datasets.

### Registry visibility 


### Registry metadata
Core registry details such as name and descriptions are not editable by users. Only an organization administrator can edit the description or name. In addition.


### Registry artifact types
Accepted artifacts types within these registries cannot be removed. However, you can add additional artifact types.

## Custom registries

### Registry visibility 

### Registry metadata

### Registry artifact types -->



