---
title: Artifact data privacy and compliance
description: W\&B 파일이 기본적으로 어디에 저장되는지 알아보세요. 민감한 정보를 저장하고 보관하는 방법을 살펴보세요.
menu:
  default:
    identifier: ko-guides-core-artifacts-data-privacy-and-compliance
    parent: artifacts
---

Artifacts를 로깅할 때 파일은 W&B에서 관리하는 Google Cloud 버킷에 업로드됩니다. 버킷의 내용은 저장 시와 전송 중에 모두 암호화됩니다. 아티팩트 파일은 해당 프로젝트에 엑세스 권한이 있는 사용자에게만 표시됩니다.

{{< img src="/images/artifacts/data_and_privacy_compliance_1.png" alt="GCS W&B Client Server diagram" >}}

아티팩트 버전을 삭제하면 데이터베이스에서 소프트 삭제로 표시되고 스토리지 비용에서 제거됩니다. 전체 아티팩트를 삭제하면 영구 삭제 대기열에 추가되고 모든 콘텐츠가 W&B 버킷에서 제거됩니다. 파일 삭제와 관련된 특정 요구 사항이 있는 경우 [고객 지원](mailto:support@wandb.com)에 문의하십시오.

멀티 테넌트 환경에 상주할 수 없는 중요한 데이터셋의 경우 클라우드 버킷에 연결된 프라이빗 W&B 서버 또는 _reference artifacts_를 사용할 수 있습니다. 레퍼런스 아티팩트는 파일 내용을 W&B로 보내지 않고 프라이빗 버킷에 대한 레퍼런스를 추적합니다. 레퍼런스 아티팩트는 버킷 또는 서버의 파일에 대한 링크를 유지 관리합니다. 즉, W&B는 파일 자체가 아닌 파일과 연결된 메타데이터만 추적합니다.

{{< img src="/images/artifacts/data_and_privacy_compliance_2.png" alt="W&B Client Server Cloud diagram" >}}

레퍼런스가 아닌 아티팩트를 만드는 방법과 유사하게 레퍼런스 아티팩트를 만듭니다.

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("animals", type="dataset")
artifact.add_reference("s3://my-bucket/animals")
```

대안이 필요하면 [contact@wandb.com](mailto:contact@wandb.com)으로 문의하여 프라이빗 클라우드 및 온프레미스 설치에 대해 문의하십시오.
