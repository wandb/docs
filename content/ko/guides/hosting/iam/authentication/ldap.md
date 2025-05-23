---
title: Configure SSO with LDAP
menu:
  default:
    identifier: ko-guides-hosting-iam-authentication-ldap
    parent: authentication
---

W&B 서버 LDAP 서버로 인증 정보를 인증합니다. 다음 가이드는 W&B 서버의 설정을 구성하는 방법을 설명합니다. 필수 및 선택적 설정과 시스템 설정 UI에서 LDAP 연결을 구성하는 방법에 대한 지침을 다룹니다. 또한 어드레스, 기본 식별 이름, 속성과 같은 LDAP 설정의 다양한 입력에 대한 정보를 제공합니다. 이러한 속성은 W&B 앱 UI 또는 환경 변수를 사용하여 지정할 수 있습니다. 익명 바인딩 또는 관리자 DN 및 비밀번호를 사용하여 바인딩을 설정할 수 있습니다.

{{% alert %}}
W&B 관리자 역할만 LDAP 인증을 활성화하고 구성할 수 있습니다.
{{% /alert %}}

## LDAP 연결 구성

{{< tabpane text=true >}}
{{% tab header="W&B App" value="app" %}}
1. W&B 앱으로 이동합니다.
2. 오른쪽 상단에서 프로필 아이콘을 선택합니다. 드롭다운에서 **시스템 설정**을 선택합니다.
3. **LDAP 클라이언트 구성**을 토글합니다.
4. 양식에 세부 정보를 추가합니다. 각 입력에 대한 자세한 내용은 **설정 파라미터** 섹션을 참조하세요.
5. **설정 업데이트**를 클릭하여 설정을 테스트합니다. 이렇게 하면 W&B 서버와 테스트 클라이언트/연결이 설정됩니다.
6. 연결이 확인되면 **LDAP 인증 활성화**를 토글하고 **설정 업데이트** 버튼을 선택합니다.
{{% /tab %}}

{{% tab header="Environment variable" value="env"%}}
다음 환경 변수를 사용하여 LDAP 연결을 설정합니다.

| 환경 변수                   | 필수 | 예                               |
| ----------------------------- | -------- | --------------------------------- |
| `LOCAL_LDAP_ADDRESS`          | 예      | `ldaps://ldap.example.com:636`   |
| `LOCAL_LDAP_BASE_DN`          | 예      | `email=mail,group=gidNumber`     |
| `LOCAL_LDAP_BIND_DN`          | 아니요    | `cn=admin`, `dc=example,dc=org`  |
| `LOCAL_LDAP_BIND_PW`          | 아니요    |                                  |
| `LOCAL_LDAP_ATTRIBUTES`       | 예      | `email=mail`, `group=gidNumber`     |
| `LOCAL_LDAP_TLS_ENABLE`       | 아니요    |                                  |
| `LOCAL_LDAP_GROUP_ALLOW_LIST` | 아니요    |                                  |
| `LOCAL_LDAP_LOGIN`            | 아니요    |                                  |

각 환경 변수의 정의는 [설정 파라미터]({{< relref path="#configuration-parameters" lang="ko" >}}) 섹션을 참조하세요. 명확성을 위해 환경 변수 접두사 `LOCAL_LDAP`이 정의 이름에서 생략되었습니다.
{{% /tab %}}
{{< /tabpane >}}

## 설정 파라미터

다음 표는 필수 및 선택적 LDAP 설정을 나열하고 설명합니다.

| 환경 변수    | 정의                                                    | 필수 |
| -------------------- | ----------------------- | -------- |
| `ADDRESS`            | 이는 W&B 서버를 호스팅하는 VPC 내의 LDAP 서버의 어드레스입니다.                                  | 예      |
| `BASE_DN`            | 루트 경로는 검색이 시작되는 위치이며 이 디렉토리로 쿼리를 수행하는 데 필요합니다.                       | 예      |
| `BIND_DN`            | LDAP 서버에 등록된 관리 사용자의 경로입니다. LDAP 서버가 인증되지 않은 바인딩을 지원하지 않는 경우에 필요합니다. 지정된 경우 W&B 서버는 이 사용자로 LDAP 서버에 연결합니다. 그렇지 않으면 W&B 서버는 익명 바인딩을 사용하여 연결합니다. | 아니요    |
| `BIND_PW`            | 관리 사용자의 비밀번호이며 바인딩을 인증하는 데 사용됩니다. 비워두면 W&B 서버는 익명 바인딩을 사용하여 연결합니다.              | 아니요    |
| `ATTRIBUTES`         | 이메일 및 그룹 ID 속성 이름을 쉼표로 구분된 문자열 값으로 제공합니다.                            | 예      |
| `TLS_ENABLE`         | TLS를 활성화합니다.                                             | 아니요    |
| `GROUP_ALLOW_LIST`   | 그룹 허용 목록입니다.                                              | 아니요    |
| `LOGIN`              | 이는 W&B 서버에 LDAP를 사용하여 인증하도록 지시합니다. `True` 또는 `False`로 설정합니다. 선택적으로 LDAP 설정을 테스트하려면 이를 false로 설정합니다. LDAP 인증을 시작하려면 이를 true로 설정합니다. | 아니요    |
