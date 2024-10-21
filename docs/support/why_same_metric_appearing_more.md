---
title: "Why is the same metric appearing more than once?"
tags:
   - experiments
---

If you're logging different types of data under the same key, we have to split them out in our database. This means you'll see multiple entries of the same metric name in a dropdown in the UI. The types we group by are `number`, `string`, `bool`, `other` (mostly arrays), and any `wandb` data type (`Histogram`, `Image`, etc). Send only one type to each key to avoid this behavior.

We store metrics in a case-insensitive fashion, so make sure you don't have two metrics with the same name like `"My-Metric"` and `"my-metric"`.