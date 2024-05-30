---
displayed_sidebar: default
---

# Registry types

W&B supports two types of registries: [Core registries](#core-registries) and [Custom registries](#custom-registries). 

## Core registries
A core registry is a template for specific use cases with a standardizes setup to maintain core functionality. W&B automatically creates core registries and adds them to the W&B Registry App. 

There are two core registries that  W&B supports: Models and Datasets. [INSERT].

## Custom registries
Use custom registries for greater flexibility and customization. [INSERT]

For information on  how to create a custom registry, see [Create a custom registry](./create_collection.md).


## Registry visibility 


There are two registry visibility types: restricted or organization visibility. 

| Visibility | Description |
| --- | --- |
| Organization | Anyone in the organization can view the registry. |
| Restricted   | Only invited organization members can view and edit the registry.| 

Core registries have organization visibility. You can not change the visibility of a core registry. 

A custom registry can have either organization or restricted visibility.  You can change the visibility of a custom registry from organization to restricted. However, you can not change a custom registry's visibility from restricted to organization visibility.

## Summary
The proceeding table summarizes the differences between core and custom registries:

|                | Core  | Custom|
| -------------- | ----- | ----- |
| Visibility     | Organizational visibility only. Visibility can not be altered. | Either organization or restricted. Visibility can be altered from organization to restricted visibility.|
| Metadata       | Preconfigured and not editable by users. | Users can edit.  |
| Artifact types | Preconfigured artifact types that cannot be removed. Users can add additional types. | Fully customizable by users.|
| Flexibility    | Limited to adding additional types to the existing list.| High flexibility. Users can customize names, descriptions, accepted artifact types and visibility.|

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



