---
title: Histogram
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/sdk/data_types/histogram.py#L18-L94 >}}

wandb class for histograms.

This object works just like numpy's histogram function
https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html

#### Examples:

Generate histogram from a sequence

```python
wandb.Histogram([1, 2, 3])
```

Efficiently initialize from np.histogram.

```python
hist = np.histogram(data)
wandb.Histogram(np_histogram=hist)
```

| Args |  |
| :--- | :--- |
|  `sequence` |  (array_like) input data for histogram |
|  `np_histogram` |  (numpy histogram) alternative input of a precomputed histogram |
|  `num_bins` |  (int) Number of bins for the histogram. The default number of bins is 64. The maximum number of bins is 512 |

| Attributes |  |
| :--- | :--- |
|  `bins` |  ([float]) edges of bins |
|  `histogram` |  ([int]) number of elements falling in each bin |

| Class Variables |  |
| :--- | :--- |
|  `MAX_LENGTH`<a id="MAX_LENGTH"></a> |  `512` |
