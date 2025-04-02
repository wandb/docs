---
title: Embed a report
description: W&B Reports を Notion に直接埋め込んだり、HTML IFrame 要素で埋め込んだりできます。
menu:
  default:
    identifier: ja-guides-core-reports-embed-reports
    parent: reports
weight: 50
---

## HTML iframe 要素

レポート の右上隅にある [**共有**] ボタンを選択します。モーダルウィンドウが表示されます。モーダルウィンドウ内で、[**埋め込みコードをコピー**] を選択します。コピーされたコードは、インラインフレーム (IFrame) HTML 要素内にレンダリングされます。コピーしたコードを、選択した iframe HTML 要素に貼り付けます。

{{% alert %}}
**公開** レポートのみが埋め込み時に表示可能です。
{{% /alert %}}

{{< img src="/images/reports/get_embed_url.gif" alt="" >}}

## Confluence

次のアニメーションは、Confluence の IFrame セル内にレポートへの直接リンクを挿入する方法を示しています。

{{< img src="//images/reports/embed_iframe_confluence.gif" alt="" >}}

## Notion

次のアニメーションは、Notion の Embed ブロックとレポートの埋め込みコードを使用して、レポートを Notion ドキュメントに挿入する方法を示しています。

{{< img src="//images/reports/embed_iframe_notion.gif" alt="" >}}

## Gradio

`gr.HTML` 要素を使用すると、Gradio Apps 内に W&B Reports を埋め込み、Hugging Face Spaces 内で使用できます。

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
