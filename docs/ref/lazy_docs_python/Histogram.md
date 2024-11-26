import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

# Histogram

<CTAButtons githubLink='https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/histogram.py'/>




## <kbd>class</kbd> `Histogram`
wandb class for histograms. 

This object works just like numpy's histogram function https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html 



**Examples:**
  Generate histogram from a sequence ```python
     wandb.Histogram([1, 2, 3])
    ``` 

 Efficiently initialize from np.histogram. ```python
     hist = np.histogram(data)
     wandb.Histogram(np_histogram=hist)
    ``` 



**Args:**
 
 - `sequence`:  (array_like) input data for histogram 
 - `np_histogram`:  (numpy histogram) alternative input of a precomputed histogram 
 - `num_bins`:  (int) Number of bins for the histogram.  The default number of bins  is 64.  The maximum number of bins is 512 



**Attributes:**
 
 - `bins`:  ([float]) edges of bins 
 - `histogram`:  ([int]) number of elements falling in each bin 

### <kbd>method</kbd> `Histogram.__init__`

```python
__init__(
    sequence: Optional[Sequence] = None,
    np_histogram: Optional[ForwardRef('NumpyHistogram')] = None,
    num_bins: int = 64
) → None
```








---

### <kbd>method</kbd> `Histogram.to_json`

```python
to_json(
    run: Optional[ForwardRef('LocalRun'), ForwardRef('Artifact')] = None
) → dict
```