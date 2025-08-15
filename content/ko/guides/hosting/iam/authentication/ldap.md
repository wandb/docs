---
title: LDAP로 SSO 구성하기
menu:
  default:
    identifier: ko-guides-hosting-iam-authentication-ldap
    parent: authentication
---

W&B 서버의 LDAP 서버와 자격 증명을 인증하세요. 아래 가이드는 W&B 서버에 대한 설정 방법을 설명합니다. 필수 및 선택적 설정 항목뿐만 아니라, 시스템 설정 UI에서 LDAP 연결을 구성하는 방법도 안내합니다. 또한 LDAP 설정의 각 입력값(예: 어드레스, base distinguished name, 속성 등)에 대한 정보도 제공합니다. 이 속성들은 W&B App UI나 환경 변수로 지정할 수 있습니다. 익명 바인딩 또는 관리자 DN과 비밀번호로 바인딩을 설정할 수 있습니다.

{{% alert %}}
LDAP 인증 활성화 및 설정은 W&B Admin 역할만 가능합니다.
{{% /alert %}}

## LDAP 연결 설정하기

{{< tabpane text=true >}}
{{% tab header="W&B App" value="app" %}}
1. W&B App으로 이동합니다.
2. 우측 상단에서 프로필 아이콘을 선택한 다음, 드롭다운에서 **System Settings**를 선택하세요.
3. **Configure LDAP Client**를 토글합니다.
4. 양식에 세부 정보를 입력합니다. 각 입력값에 대한 자세한 내용은 **Configuring Parameters** 섹션을 참고하세요.
5. **Update Settings**를 클릭해 설정을 테스트합니다. 이것은 W&B 서버와의 테스트 클라이언트/연결을 생성하는 과정입니다.
6. 연결이 검증되면, **Enable LDAP Authentication**을 토글하고 **Update Settings** 버튼을 선택하세요.
{{% /tab %}}

{{% tab header="Environment variable" value="env"%}}
다음 환경 변수로 LDAP 연결을 설정할 수 있습니다:

| 환경 변수                  | 필수 여부 | 예시                            |
| -------------------------- | -------- | ------------------------------- |
| `LOCAL_LDAP_ADDRESS`       | 예       | `ldaps://ldap.example.com:636`  |
| `LOCAL_LDAP_BASE_DN`       | 예       | `email=mail,group=gidNumber`    |
| `LOCAL_LDAP_BIND_DN`       | 아니오   | `cn=admin`, `dc=example,dc=org` |
| `LOCAL_LDAP_BIND_PW`       | 아니오   |                                 |
| `LOCAL_LDAP_ATTRIBUTES`    | 예       | `email=mail`, `group=gidNumber` |
| `LOCAL_LDAP_TLS_ENABLE`    | 아니오   |                                 |
| `LOCAL_LDAP_GROUP_ALLOW_LIST`| 아니오 |                                 |
| `LOCAL_LDAP_LOGIN`         | 아니오   |                                 |

각 환경 변수의 정의는 [설정 파라미터]({{< relref path="#configuration-parameters" lang="ko" >}}) 섹션을 참고하세요. 참고로, 명확성을 위해 환경 변수 정의에서는 `LOCAL_LDAP` 접두어를 생략했습니다.
{{% /tab %}}
{{< /tabpane >}}

## 설정 파라미터

아래 표는 필수 및 선택 가능한 LDAP 설정값에 대해 설명합니다.

| 환경 변수        | 정의                                                                                       | 필수 여부 |
| ---------------- | ----------------------------------------------------------------------------------------- | -------- |
| `ADDRESS`        | W&B Server가 호스팅되는 VPC 내에서 사용하는 LDAP 서버의 어드레스입니다.                     | 예       |
| `BASE_DN`        | LDAP 디렉토리에서 검색이 시작되는 루트 경로입니다. 쿼리를 실행할 때 이 값이 필요합니다.          | 예       |
| `BIND_DN`        | LDAP 서버에 등록된 관리자의 경로입니다. LDAP 서버가 인증 없는 바인딩을 지원하지 않는 경우 필수입니다. 지정하면 W&B Server는 이 사용자로 LDAP 서버에 연결합니다. 그렇지 않으면 익명 바인딩으로 연결합니다. | 아니오   |
| `BIND_PW`        | 관리자 사용자 인증에 사용하는 비밀번호입니다. 비워두면 W&B Server는 익명 바인딩을 사용해 연결합니다. | 아니오   |
| `ATTRIBUTES`     | 이메일과 그룹 ID 속성명을 콤마로 구분해 입력하세요.                                          | 예       |
| `TLS_ENABLE`     | TLS 사용 여부를 설정합니다.                                                                 | 아니오   |
| `GROUP_ALLOW_LIST`| 그룹 허용 리스트입니다.                                                                    | 아니오   |
| `LOGIN`          | W&B Server에서 LDAP를 인증에 사용할지 지정합니다. `True` 또는 `False`로 설정하세요. LDAP 설정을 테스트할 때는 임시로 false로 둘 수 있습니다. LDAP 인증을 시작하려면 true로 변경하세요. | 아니오   |