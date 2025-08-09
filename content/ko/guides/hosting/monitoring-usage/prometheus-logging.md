---
title: Prometheus 모니터링 사용하기
menu:
  default:
    identifier: ko-guides-hosting-monitoring-usage-prometheus-logging
    parent: monitoring-and-usage
weight: 2
---

[Prometheus](https://prometheus.io/docs/introduction/overview/)를 W&B Server와 함께 사용하세요. Prometheus는 [kubernetes ClusterIP 서비스](https://github.com/wandb/terraform-kubernetes-wandb/blob/main/main.tf#L225)로 노출됩니다.

{{% alert color="secondary" %}}
Prometheus 모니터링은 [셀프 관리 인스턴스]({{< relref path="/guides/hosting/hosting-options/self-managed.md" lang="ko" >}})에서만 사용할 수 있습니다.
{{% /alert %}}

아래 절차를 따라 Prometheus 메트릭 엔드포인트(`/metrics`)에 엑세스하세요.

1. Kubernetes CLI 툴킷인 [kubectl](https://kubernetes.io/docs/reference/kubectl/)로 클러스터에 연결하세요. 자세한 내용은 kubernetes의 [클러스터 엑세스](https://kubernetes.io/docs/tasks/access-application-cluster/access-cluster/) 문서를 참고하세요.
2. 다음 명령어로 클러스터의 내부 어드레스를 확인하세요.

    ```bash
    kubectl describe svc prometheus
    ```

3. Kubernetes 클러스터에서 실행 중인 컨테이너 내부에서 셸 세션을 시작하려면 [`kubectl exec`](https://kubernetes.io/docs/reference/generated/kubectl/kubectl-commands)를 사용하세요. 엔드포인트 `<internal address>/metrics`에 접속하세요.

   아래 코맨드를 복사해 터미널에 붙여넣고 `<internal address>`를 실제 내부 어드레스로 바꿔 실행하세요.

   ```bash
   kubectl exec <internal address>/metrics
   ```

테스트용 pod가 시작되며, 네트워크 내 어떤 곳에든 엑세스할 수 있도록 exec로 직접 들어갈 수 있습니다.

```bash
kubectl run -it testpod --image=alpine bin/ash --restart=Never --rm
```

이후 네트워크 내부에서만 엑세스를 유지하거나, 원한다면 kubernetes nodeport 서비스를 이용해 직접 외부로 노출할 수도 있습니다.