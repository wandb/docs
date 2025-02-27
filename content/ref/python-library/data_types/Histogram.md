---
title: Histogram
object_type: data_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/sdk/data_types/histogram.py >}}




## <kbd>class</kbd> `Histogram`
W&B class for histograms. 

This object works just like numpy's histogram function https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html 





**Args:**
 
 - `sequence`:  Input data for histogram. 
 - `np_histogram`:  Alternative input of a precomputed histogram. 
 - `num_bins`:  Number of bins for the histogram.  The default number of bins  is 64. The maximum number of bins is 512. 



**Attributes:**
 
 - `bins` ([float]):  Edges of bins 
 - `histogram` ([int]):  Number of elements falling in each bin. 



**Examples:**
 Generate histogram from a sequence. 

```python
import wandb

wandb.Histogram([1, 2, 3])
``` 

Efficiently initialize from np.histogram. 

```python
import numpy as np
import wandb

hist = np.histogram(data)
wandb.Histogram(np_histogram=hist)
``` 

### <kbd>method</kbd> `Histogram.__init__`

```python
__init__(
    sequence: Optional[Sequence] = None,
    np_histogram: Optional[ForwardRef('NumpyHistogram')] = None,
    num_bins: int = 64
) → None
```








---

