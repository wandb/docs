---
title: Dockerfile を指定して、W&B に Docker イメージのビルドを任せられますか？
menu:
  launch:
    identifier: ja-launch-launch-faq-dockerfile_let_wb_build_docker_image_me
    parent: launch-faq
---

この機能は、要件は安定しているが コードベース が頻繁に変わる プロジェクト に適しています。

{{% alert color="secondary" %}}
Dockerfile をマウントを利用するように記述してください。詳細は、[Docker Docs ウェブサイトの Mounts ドキュメント](https://docs.docker.com/build/guide/mounts/)を参照してください。
{{% /alert %}}

Dockerfile を設定したら、W&B への指定方法は次の 3 通りです:

* Dockerfile.wandb を使う
* W&B CLI を使う
* W&B App を使う

{{< tabpane text=true >}}
{{% tab "Dockerfile.wandb" %}}
W&B の run の エントリポイントと同じ ディレクトリーに `Dockerfile.wandb` ファイルを配置してください。W&B は組み込みの Dockerfile の代わりにこのファイルを使用します。 
{{% /tab %}}
{{% tab "W&B CLI" %}}
`wandb launch` コマンドで `--dockerfile` フラグを使って ジョブ をキューに追加します:

```bash
wandb launch --dockerfile path/to/Dockerfile
```
{{% /tab %}}
{{% tab "W&B app" %}}
W&B App でキューに ジョブ を追加する際、Dockerfile のパスを **Overrides** セクションで指定します。キーと 値 のペアとして入力し、キーは `"dockerfile"`、値は Dockerfile へのパスにします。

次の JSON は、ローカル ディレクトリーにある Dockerfile を含める方法を示しています:

```json title="W&B App の Launch ジョブ"
{
  "args": [],
  "run_config": {
    "lr": 0,
    "batch_size": 0,
    "epochs": 0
  },
  "entrypoint": [],
  "dockerfile": "./Dockerfile"
}
```
{{% /tab %}}
{{% /tabpane %}}