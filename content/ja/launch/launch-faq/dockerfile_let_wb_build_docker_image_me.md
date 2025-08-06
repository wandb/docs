---
title: Dockerfile を指定して、W&B に Docker イメージをビルドしてもらうことはできますか？
menu:
  launch:
    identifier: dockerfile_let_wb_build_docker_image_me
    parent: launch-faq
---

この機能は、要件が安定しているものの、コードベースが頻繁に変更される Projects に最適です。

{{% alert color="secondary" %}}
Dockerfile をマウントを使用するようにフォーマットしてください。詳細については、[Docker Docs サイトの Mounts ドキュメント](https://docs.docker.com/build/guide/mounts/)をご覧ください。
{{% /alert %}}

Dockerfile の設定後、W&B に次の 3 つの方法のいずれかで指定します。

* Dockerfile.wandb を使用
* W&B CLI を使用
* W&B App を使用

{{< tabpane text=true >}}
{{% tab "Dockerfile.wandb" %}}
W&B run のエントリーポイントと同じディレクトリーに `Dockerfile.wandb` ファイルを配置してください。W&B は、このファイルを組み込みの Dockerfile の代わりに使用します。
{{% /tab %}}
{{% tab "W&B CLI" %}}
`wandb launch` コマンドで `--dockerfile` フラグを使ってジョブをキューに追加します：

```bash
wandb launch --dockerfile path/to/Dockerfile
```
{{% /tab %}}
{{% tab "W&B app" %}}
W&B App でジョブをキューに追加する場合、**Overrides** セクションで Dockerfile のパスを指定します。`"dockerfile"` をキー、Dockerfile のパスを値とする key-value ペアで入力してください。

以下の JSON は、ローカルディレクトリー内の Dockerfile を含める例です。

```json title="Launch job W&B App"
// ローンンチジョブを W&B App で指定する例です
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