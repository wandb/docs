import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

# Graph

<CTAButtons githubLink='https://github.com/wandb/wandb/blob/main/wandb/data_types.py'/>




## <kbd>class</kbd> `Graph`
Wandb class for graphs. 

This class is typically used for saving and displaying neural net models.  It represents the graph as an array of nodes and edges.  The nodes can have labels that can be visualized by wandb. 



**Examples:**
  Import a keras model: ```
     Graph.from_keras(keras_model)
    ``` 



**Attributes:**
 
 - `format` (string):  Format to help wandb display the graph nicely. 
 - `nodes` ([wandb.Node]):  List of wandb.Nodes 
 - `nodes_by_id` (dict):  dict of ids -> nodes 
 - `edges` ([(wandb.Node, wandb.Node)]):  List of pairs of nodes interpreted as edges 
 - `loaded` (boolean):  Flag to tell whether the graph is completely loaded 
 - `root` (wandb.Node):  root node of the graph 

### <kbd>method</kbd> `Graph.__init__`

```python
__init__(format='keras')
```








---

### <kbd>method</kbd> `Graph.add_edge`

```python
add_edge(from_node, to_node)
```





---

### <kbd>method</kbd> `Graph.add_node`

```python
add_node(node=None, **node_kwargs)
```





---

### <kbd>method</kbd> `Graph.bind_to_run`

```python
bind_to_run(*args, **kwargs)
```





---

### <kbd>classmethod</kbd> `Graph.from_keras`

```python
from_keras(model)
```





---

### <kbd>classmethod</kbd> `Graph.get_media_subdir`

```python
get_media_subdir()
```





---

### <kbd>method</kbd> `Graph.pprint`

```python
pprint()
```





---

### <kbd>method</kbd> `Graph.to_json`

```python
to_json(run)
```