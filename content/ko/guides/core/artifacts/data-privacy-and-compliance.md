---
title: Artifacts 데이터 프라이버시 및 컴플라이언스
description: W&B 파일이 기본적으로 어디에 저장되는지 확인하고, 파일 저장 및 민감한 정보 관리 방법을 알아보세요.
menu:
  default:
    identifier: ko-guides-core-artifacts-data-privacy-and-compliance
    parent: artifacts
---

파일들은 Artifacts를 로그할 때 W&B에서 관리하는 Google Cloud 버킷에 업로드됩니다. 버킷의 모든 내용은 저장 시와 전송 중 모두 암호화됩니다. Artifact 파일은 해당 Projects에 엑세스 권한이 있는 Users만 볼 수 있습니다.

{{< img src="/images/artifacts/data_and_privacy_compliance_1.png" alt="GCS W&B Client Server diagram" >}}

아티팩트의 버전을 삭제하면, 데이터베이스에 소프트 삭제로 표시되고 스토리지 요금에서 제외됩니다. 전체 아티팩트를 삭제하면 영구 삭제를 위해 큐에 추가되며, 해당 콘텐츠는 모두 W&B 버킷에서 제거됩니다. 파일 삭제와 관련해 특별한 요구 사항이 있다면 [Customer Support](mailto:support@wandb.com)로 문의해 주세요.

여러 사용자가 공유하는 환경에 둘 수 없는 민감한 Datasets의 경우, 클라우드 버킷에 연결된 프라이빗 W&B 서버 또는 _reference artifacts_ 를 사용할 수 있습니다. Reference artifacts는 W&B로 파일 내용을 전송하지 않고 개인 버킷에 대한 참조만을 추적합니다. Reference artifacts는 버킷이나 서버의 파일에 대한 링크만을 유지합니다. 즉, W&B는 파일 자체가 아니라 파일과 연관된 메타데이터만을 관리합니다.

{{< img src="/images/artifacts/data_and_privacy_compliance_2.png" alt="W&B Client Server Cloud diagram" >}}

Reference artifact는 일반 artifact를 생성하는 것과 비슷하게 만들 수 있습니다:

```python
import wandb

run = wandb.init()
artifact = wandb.Artifact("animals", type="dataset")
artifact.add_reference("s3://my-bucket/animals")
# "animals"라는 이름의 dataset 타입 reference artifact 생성
# "s3://my-bucket/animals" 경로를 reference로 추가
```

기타 대안이 필요하시다면, [contact@wandb.com](mailto:contact@wandb.com) 으로 연락하셔서 프라이빗 클라우드 및 온프레미스 설치에 대해 상담해 주세요.