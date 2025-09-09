---
title: Prometheus を使用した監視
menu:
  default:
    identifier: ja-guides-hosting-monitoring-usage-prometheus-logging
    parent: monitoring-and-usage
weight: 2
---

W&B サーバーで [Prometheus](https://prometheus.io/docs/introduction/overview/) を使用できます。Prometheus は [Kubernetes の ClusterIP サービス](https://github.com/wandb/terraform-kubernetes-wandb/blob/main/main.tf#L225) として公開されます。
{{% alert color="secondary" %}}
Prometheus の監視は、[セルフマネージド インスタンス]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ja" >}}) でのみ利用可能です。
{{% /alert %}}

Prometheus の メトリクス エンドポイント (`/metrics`) に アクセス するには、次の手順に従ってください。

1. Kubernetes の CLI ツールである [kubectl](https://kubernetes.io/docs/reference/kubectl/) を使用して クラスター に接続します。詳細は、Kubernetes の [クラスターへのアクセス](https://kubernetes.io/docs/tasks/access-application-cluster/access-cluster/) ドキュメントを参照してください。
2. 次の コマンド で、クラスター の内部 アドレス を確認します。
    ```bash
    kubectl describe svc prometheus
    ```
3. [`kubectl exec`](https://kubernetes.io/docs/reference/generated/kubectl/kubectl-commands) を使って、Kubernetes クラスター 上で動作中のコンテナ内にシェル セッションを開始し、`<internal address>/metrics` の エンドポイント に アクセス します。
   次の コマンド をコピーして ターミナル で実行し、`<internal address>` を内部 アドレス に置き換えてください。
   ```bash
   kubectl exec <internal address>/metrics
   ```

テスト用の Pod が起動し、その中で exec を実行して ネットワーク 内の任意のリソースに アクセス できます。
```bash
kubectl run -it testpod --image=alpine bin/ash --restart=Never --rm
```
そこから、ネットワーク 内部への アクセス を維持するか、Kubernetes の NodePort サービスを使用して自分で公開するかを選択できます。