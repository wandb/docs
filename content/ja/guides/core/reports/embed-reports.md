---
title: レポートを埋め込む
description: W&B レポートを Notion に直接埋め込むか、HTML の IFrame 要素を使って埋め込むことができます。
menu:
  default:
    identifier: ja-guides-core-reports-embed-reports
    parent: reports
weight: 50
---

## HTML iframe 要素

レポート右上の **Share** ボタンを選択します。モーダルウィンドウが表示されます。モーダルウィンドウ内で **Copy embed code** を選択してください。コピーされたコードは、Inline Frame（IFrame）HTML要素内で表示されます。お好きな iframe HTML 要素にコピーしたコードを貼り付けてください。

{{% alert %}}
**public** レポートのみ埋め込み表示が可能です。
{{% /alert %}}

{{< img src="/images/reports/get_embed_url.gif" alt="埋め込みコードの取得" >}}

## Confluence

以下のアニメーションは、Confluence内のIFrameセルにレポートの直接リンクを挿入する方法を示しています。

{{< img src="//images/reports/embed_iframe_confluence.gif" alt="Confluence への埋め込み" >}}

## Notion

以下のアニメーションは、Notion の埋め込みブロックとレポートの埋め込みコードを使って、Notion ドキュメントにレポートを挿入する方法を示しています。

{{< img src="//images/reports/embed_iframe_notion.gif" alt="Notion への埋め込み" >}}

## Gradio

`gr.HTML` 要素を使って、W&B Reports を Gradio アプリ内に埋め込み、Hugging Face Spaces で利用できます。

```python
import gradio as gr


def wandb_report(url):
    # W&Bレポートをiframeで埋め込み表示
    iframe = f'<iframe src={url} style="border:none;height:1024px;width:100%">'
    return gr.HTML(iframe)


with gr.Blocks() as demo:
    report = wandb_report(
        "https://wandb.ai/_scott/pytorch-sweeps-demo/reports/loss-22-10-07-16-00-17---VmlldzoyNzU2NzAx"
    )
demo.launch()
```

##