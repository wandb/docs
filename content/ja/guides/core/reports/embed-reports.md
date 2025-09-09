---
title: Reports を埋め込む
description: W&B Reports を Notion に直接埋め込むか、HTML の IFrame 要素を使用して埋め込む。
menu:
  default:
    identifier: ja-guides-core-reports-embed-reports
    parent: reports
weight: 50
---

## HTML iframe 要素

Reports の右上隅にある「 **共有** 」ボタンを選択します。モーダルウィンドウが表示されます。モーダルウィンドウ内で、「 **埋め込みコードをコピー** 」を選択します。コピーされたコードは、Inline Frame (IFrame) HTML 要素内にレンダリングされます。コピーしたコードを、任意の iframe HTML 要素に貼り付けます。

{{% alert %}}
埋め込み表示できるのは、**公開** の Reports のみです。
{{% /alert %}}

{{< img src="/images/reports/get_embed_url.gif" alt="埋め込みコードの取得" >}}

## Confluence

以下の動画は、Confluence の IFrame セル内に Reports への直接リンクを挿入する方法を示しています。

{{< img src="//images/reports/embed_iframe_confluence.gif" alt="Confluence への埋め込み" >}}

## Notion

以下の動画は、Notion の Embed ブロックと Reports の埋め込みコードを使用して、Reports を Notion ドキュメントに挿入する方法を示しています。

{{< img src="//images/reports/embed_iframe_notion.gif" alt="Notion への埋め込み" >}}

## Gradio

`gr.HTML` 要素を使用して、Gradio アプリ内に W&B Reports を埋め込み、Hugging Face Spaces 内で使用できます。

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