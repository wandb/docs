---
description: Embed W&B reports directly into Notion or with an HTML IFrame element.
menu:
  default:
    identifier: embed-reports
    parent: reports
title: Embed a report
weight: 50
---

## HTML iframe element

Select the **Share** button on the upper right hand corner within a report. A modal window will appear. Within the modal window, select **Copy embed code**. The copied code will render within an Inline Frame (IFrame)  HTML element. Paste the copied code into an iframe HTML element of your choice.

{{% alert %}}
Only **public** reports are viewable when embedded.
{{% /alert %}}

{{< img src="/images/reports/get_embed_url.gif" alt="Getting embed code" >}}

## Confluence

The proceeding animation demonstrates how to insert the direct link to the report within an IFrame cell in Confluence.

{{< img src="//images/reports/embed_iframe_confluence.gif" alt="Embedding in Confluence" >}}

## Notion

The proceeding animation demonstrates how to insert a report into a Notion document using an Embed block in Notion and the report's embedded code.

{{< img src="//images/reports/embed_iframe_notion.gif" alt="Embedding in Notion" >}}

## Gradio

You can use the `gr.HTML` element to embed W&B Reports within Gradio Apps and use them within Hugging Face Spaces.

```python
import gradio as gr


def wandb_report(url):
    iframe = f'<iframe src={url} style="border:none;height:1024px;width:100%">'
    return gr.HTML(iframe)


with gr.Blocks() as demo:
    report = wandb_report(
        "https://wandb.ai/_scott/pytorch-sweeps-demo/reports/loss-22-10-07-16-00-17---VmlldzoyNzU2NzAx"
    )
demo.launch()
```

##