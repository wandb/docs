---
title: Embed a report
description: W&B の Reports を Notion に直接埋め込んだり、HTML の IFrame 要素を使って埋め込んだりできます。
menu:
  default:
    identifier: ja-guides-core-reports-embed-reports
    parent: reports
weight: 50
---

## HTML iframe 要素

レポート 右上にある「**共有**」ボタンを選択します。モーダルウィンドウが表示されます。モーダルウィンドウ内で、「**埋め込みコードをコピー**」を選択します。コピーされたコードは、インラインフレーム（IFrame）HTML要素内に表示されます。コピーしたコードを、任意の iframe HTML 要素に貼り付けます。

{{% alert %}}
**公開** レポート のみが、埋め込まれた状態で表示可能です。
{{% /alert %}}

{{< img src="/images/reports/get_embed_url.gif" alt="" >}}

## Confluence

次のアニメーションは、Confluence の IFrame セル内に、レポート への直接リンクを挿入する方法を示しています。

{{< img src="//images/reports/embed_iframe_confluence.gif" alt="" >}}

## Notion

次のアニメーションは、Notion のドキュメントに、Notion の埋め込みブロックと レポート の埋め込みコードを使用して、 レポート を挿入する方法を示しています。

{{< img src="//images/reports/embed_iframe_notion.gif" alt="" >}}

## Gradio

`gr.HTML` 要素を使用すると、Gradio Apps 内に W&B Reports を埋め込み、Hugging Face Spaces 内で使用できます。

```python
import gradio as gr


def wandb_report(url):
    # URL を受け取り、指定された URL の iframe を含む HTML オブジェクトを返す
    iframe = f'<iframe src={url} style="border:none;height:1024px;width:100%">'
    return gr.HTML(iframe)


with gr.Blocks() as demo:
    # wandb_report 関数を使用して レポート を埋め込んだ Gradio デモを作成
    report = wandb_report(
        "https://wandb.ai/_scott/pytorch-sweeps-demo/reports/loss-22-10-07-16-00-17---VmlldzoyNzU2NzAx"
    )
demo.launch()
```

##
