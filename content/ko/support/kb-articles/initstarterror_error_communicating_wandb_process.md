---
menu:
  support:
    identifier: ko-support-kb-articles-initstarterror_error_communicating_wandb_process
support:
- experiments
title: 'InitStartError: Error communicating with wandb process'
toc_hide: true
type: docs
url: /support/:filename
---

This error indicates that the library encounters an issue launching the process that synchronizes data to the server.

The following workarounds resolve the issue in specific environments:

{{< tabpane text=true >}}
{{% tab "Linux and OS X" %}}
```python
wandb.init(settings=wandb.Settings(start_method="fork"))
```

{{% /tab %}}
{{% tab "Google Colab" %}}

For versions prior to `0.13.0`, use:

```python
wandb.init(settings=wandb.Settings(start_method="thread"))
```
{{% /tab %}}
{{< /tabpane >}}