---
title: Graph
object_type: data_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/sdk/data_types/graph.py >}}




## <kbd>class</kbd> `Graph`
W&B class for graphs. 

This class is typically used for saving and displaying neural net models. It represents the graph as an array of nodes and edges. The nodes can have labels that can be visualized by wandb. 



**Attributes:**
 
 - `format` (string):  Format to help wandb display the graph nicely. 
 - `nodes` ([wandb.Node]):  List of `wandb.Nodes`. 
 - `nodes_by_id` (dict):  dict of ids -> nodes 
 - `edges` ([(wandb.Node, wandb.Node)]):  List of pairs of nodes interpreted  as edges. 
 - `loaded` (boolean):  Flag to tell whether the graph is completely loaded. 
 - `root` (wandb.Node):  Root node of the graph. 



**Examples:**
 

Import a keras model 

```python
import wandb

wandb.Graph.from_keras(keras_model)
``` 

### <kbd>method</kbd> `Graph.__init__`

```python
__init__(format='keras')
```








---







