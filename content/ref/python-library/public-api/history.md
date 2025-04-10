---
title: history
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/apis/public/history.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API for Run History. 

This module provides classes for efficiently scanning and sampling run history data. Classes include: 



HistoryScan: Iterator for scanning complete run history 
- Paginated access to all metrics 
- Configure step ranges and page sizes 
- Raw access to all logged data 

SampledHistoryScan: Iterator for sampling run history data 
- Efficient access to downsampled metrics 
- Filter by specific keys 
- Control sample size and step ranges 



**Note:**

> This module is part of the W&B Public API and provides methods to access run history data. It handles pagination automatically and offers both complete and sampled access to metrics logged during training runs. 



---

## <kbd>class</kbd> `HistoryScan`
Iterator for scanning complete run history. 



**Args:**
 
 - `client`:  (`wandb.apis.internal.Api`) The client instance to use 
 - `run`:  (`wandb.sdk.internal.Run`) The run object to scan history for 
 - `min_step`:  (int) The minimum step to start scanning from 
 - `max_step`:  (int) The maximum step to scan up to 
 - `page_size`:  (int) Number of samples per page (default is 1000) 

### <kbd>method</kbd> `HistoryScan.__init__`

```python
__init__(client, run, min_step, max_step, page_size=1000)
```








---


## <kbd>class</kbd> `SampledHistoryScan`
Iterator for sampling run history data. 



**Args:**
 
 - `client`:  (`wandb.apis.internal.Api`) The client instance to use 
 - `run`:  (`wandb.sdk.internal.Run`) The run object to sample history from 
 - `keys`:  (list) List of keys to sample from the history 
 - `min_step`:  (int) The minimum step to start sampling from 
 - `max_step`:  (int) The maximum step to sample up to 
 - `page_size`:  (int) Number of samples per page (default is 1000) 

### <kbd>method</kbd> `SampledHistoryScan.__init__`

```python
__init__(client, run, keys, min_step, max_step, page_size=1000)
```








---

