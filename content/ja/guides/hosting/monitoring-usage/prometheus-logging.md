---
title: Prometheus モニタリングを利用する
menu:
  default:
    identifier: ja-guides-hosting-monitoring-usage-prometheus-logging
    parent: monitoring-and-usage
weight: 2
---

[Prometheus](https://prometheus.io/docs/introduction/overview/) を W&B サーバーと組み合わせて利用できます。Prometheus は [kubernetes ClusterIP サービス](https://github.com/wandb/terraform-kubernetes-wandb/blob/main/main.tf#L225)として公開されています。

{{% alert color="secondary" %}}
Prometheus モニタリングは [セルフマネージドインスタンス]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) でのみ利用可能です。
{{% /alert %}}

以下の手順に従って、Prometheus のメトリクスエンドポイント（`/metrics`）にアクセスしてください。

1. Kubernetes CLI ツールキットの [kubectl](https://kubernetes.io/docs/reference/kubectl/) を使ってクラスターに接続します。詳細は kubernetes の [クラスターへのアクセス](https://kubernetes.io/docs/tasks/access-application-cluster/access-cluster/) ドキュメントを参照してください。
2. 次のコマンドでクラスターの内部アドレスを確認します：

    ```bash
    kubectl describe svc prometheus
    ```

3. Kubernetes クラスター内で実行中のコンテナで [`kubectl exec`](https://kubernetes.io/docs/reference/generated/kubectl/kubectl-commands) を使いシェルセッションを開始します。エンドポイント `<internal address>/metrics` に接続します。

   以下のコマンドをターミナルで実行し、`<internal address>` をあなたの内部アドレスに置き換えてください：

   ```bash
   kubectl exec <internal address>/metrics
   ```

テスト用のポッドを起動し、ネットワーク内の任意のリソースへアクセスしたい場合は以下のように exec できます：

```bash
kubectl run -it testpod --image=alpine bin/ash --restart=Never --rm
```

ここからネットワーク内部のアクセスを維持するか、kubernetes の nodeport サービスで自分で外部に公開することもできます。