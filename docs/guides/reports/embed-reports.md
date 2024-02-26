---
description: >-
  Embed Weights & Biases reports directly into Notion or with an HTML IFrame
  element.
displayed_sidebar: default
---

# レポートの埋め込み

<head>
  <title>人気アプリケーションにレポートを埋め込む</title>
</head>

### HTMLのiframe要素

レポート内の右上にある**共有**ボタンを選択します。モーダルウィンドウが表示されます。モーダルウィンドウ内で「埋め込み用コードをコピー」を選択します。コピーされたコードは、インラインフレーム（IFrame）HTML要素内にレンダリングされます。コピーされたコードをお選びのiframe HTML要素に貼り付けてください。

_注意: 現在埋め込まれた状態で表示可能なのは**公開**されているレポートのみです。_

__

![](@site/static/images/reports/get_embed_url.gif)

### Confluence

以下のアニメーションは、ConfluenceのIFrameセルにレポートへの直接リンクを挿入する方法を示しています。

![](@site/static/images/reports/embed_iframe_confluence.gif)

### Notion

以下のアニメーションは、NotionのEmbedブロックとレポートの埋め込みコードを使用して、Notionドキュメントにレポートを挿入する方法を示しています。
![](@site/static/images/reports/embed_iframe_notion.gif)

### Gradio

Gradioアプリケーション内にW&Bレポートを埋め込んで、Hugging Face Spacesで使用するために、`gr.HTML`要素を使用できます。

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