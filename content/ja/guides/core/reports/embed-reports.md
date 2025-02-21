---
title: Embed a report
description: Notion または HTML IFrame 要素を使用して、W&B レポートを直接埋め込むことができます。
menu:
  default:
    identifier: ja-guides-core-reports-embed-reports
    parent: reports
weight: 50
---

## HTML iframe element

レポート内の右上隅にある **Share** ボタンを選択します。モーダルウィンドウが表示されます。モーダルウィンドウ内で **Copy embed code** を選択します。コピーしたコードは、インラインフレーム (IFrame) HTML 要素内で表示されます。コピーしたコードを自分の選んだ iframe HTML 要素に貼り付けてください。

{{% alert %}}
埋め込んで表示できるのは **public** レポートのみです。
{{% /alert %}}

{{< img src="/images/reports/get_embed_url.gif" alt="" >}}

## Confluence

次のアニメーションは、IFrame セル内でレポートへの直接リンクを Confluence に挿入する方法を示しています。

{{< img src="//images/reports/embed_iframe_confluence.gif" alt="" >}}

## Notion

次のアニメーションは、Notion の Embed ブロックとレポートの埋め込みコードを使用して、レポートを Notion ドキュメントに挿入する方法を示しています。

{{< img src="//images/reports/embed_iframe_notion.gif" alt="" >}}

## Gradio

`gr.HTML` 要素を使用して、W&B Reports を Gradio Apps に埋め込み、Hugging Face Spaces 内で使用することができます。

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