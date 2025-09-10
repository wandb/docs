---
title: OnAddArtifactAlias
object_type: automations_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/events.py >}}



## <kbd>class</kbd> `OnAddArtifactAlias`
A new alias is assigned to an artifact.


**Args:**
 
 - `event_type` (Literal[ADD_ARTIFACT_ALIAS]): 
 - `scope` (Union[_ArtifactSequenceScope, _ArtifactPortfolioScope, ProjectScope]): The scope of the event.
 - `filter` (Union[And, Or, Nor, Not, Lt, Gt, Lte, Gte, Eq, Ne, In, NotIn, Exists, Regex, Contains, Dict[str, Any], FilterExpr]): Additional condition(s), if any, that must be met for this event to trigger an automation.

**Returns:**
 An `OnAddArtifactAlias` object.

### <kbd>method</kbd> `OnAddArtifactAlias.__init__`

```python
__init__(
    event_type: 'Literal[ADD_ARTIFACT_ALIAS]' = ADD_ARTIFACT_ALIAS,
    scope: '_ArtifactSequenceScope | _ArtifactPortfolioScope | ProjectScope',
    filter: 'And | Or | Nor | Not | Lt | Gt | Lte | Gte | Eq | Ne | In | NotIn | Exists | Regex | Contains | dict[str, Any] | FilterExpr' = And([])
) → None
```
