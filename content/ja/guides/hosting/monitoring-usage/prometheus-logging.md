---
title: Prometheus モニタリングを使用する
menu:
  default:
    identifier: ja-guides-hosting-monitoring-usage-prometheus-logging
    parent: monitoring-and-usage
weight: 2
---

[Prometheus](https://prometheus.io/docs/introduction/overview/) を W&B サーバーと一緒に使用します。Prometheus のインストールは [Kubernetes ClusterIP サービス](https://github.com/wandb/terraform-kubernetes-wandb/blob/main/main.tf#L225) として公開されます。

{{% alert color="secondary" %}}
Prometheus モニタリングは [セルフマネージドインスタンス]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) でのみ利用可能です。
{{% /alert %}}

以下の手順に従って、Prometheus メトリクスエンドポイント (`/metrics`) にアクセスします：

1. Kubernetes CLI ツールキットの [kubectl](https://kubernetes.io/docs/reference/kubectl/) を使用してクラスターに接続します。詳細については、Kubernetes の [クラスターへのアクセス](https://kubernetes.io/docs/tasks/access-application-cluster/access-cluster/) ドキュメントを参照してください。
2. クラスターの内部アドレスを見つけます：

    ```bash
    kubectl describe svc prometheus
    ```

3. Kubernetes クラスターで実行中のコンテナ内でシェルセッションを開始し、[`kubectl exec`](https://kubernetes.io/docs/reference/generated/kubectl/kubectl-commands) を使用して、`<internal address>/metrics` エンドポイントにアクセスします。

   以下のコマンドをコピーしてターミナルで実行し、`<internal address>` を内部アドレスに置き換えます：

   ```bash
   kubectl exec <internal address>/metrics
   ```

テストポッドが開始され、ネットワーク内の何かにアクセスするためだけに exec することができます：

```bash
kubectl run -it testpod --image=alpine bin/ash --restart=Never --rm
```

そこから、ネットワークへのアクセスを内部に保つか、kubernetes nodeport サービスで自分で公開するかを選択できます。