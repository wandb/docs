---
title: 'teams

  '
data_type_classification: module
menu:
  reference:
    identifier: ko-ref-python-public-api-teams
object_type: public_apis_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/apis/public/teams.py >}}




# <kbd>module</kbd> `wandb.apis.public`
W&B Public API 를 사용한 팀 및 팀 멤버 관리

이 모듈은 W&B Teams 와 그 멤버를 관리하는 클래스를 제공합니다.



**참고:**

> 이 모듈은 W&B Public API 의 일부이며, 팀 및 팀 멤버를 관리하는 메소드들을 제공합니다. 팀 관리 작업을 하려면 적절한 권한이 필요합니다. 



---

## <kbd>class</kbd> `Member`
팀의 멤버를 나타내는 클래스입니다.



**ARG:**
 
 - `client` (`wandb.apis.internal.Api`):  사용할 클라이언트 인스턴스
 - `team` (str):  이 멤버가 속한 팀의 이름
 - `attrs` (dict):  멤버 속성



### <kbd>method</kbd> `Member.__init__`

```python
__init__(client, team, attrs)
```








---

### <kbd>method</kbd> `Member.delete`

```python
delete()
```

팀에서 멤버를 제거합니다.



**반환값:**
  성공 여부를 나타내는 Boolean 값


---

## <kbd>class</kbd> `Team`
W&B 팀을 나타내는 클래스입니다.

이 클래스는 W&B 팀을 관리하기 위한 다양한 메소드를 제공합니다. 예를 들어, 팀 생성, 멤버 초대, 서비스 계정 관리 등이 있습니다. 팀 속성 처리를 위해 Attrs 를 상속합니다.



**ARG:**
 
 - `client` (`wandb.apis.public.Api`):  사용할 api 인스턴스
 - `name` (str):  팀의 이름
 - `attrs` (dict):  (옵션) 팀 속성 사전



**참고:**

> 팀 관리에는 적절한 권한이 필요합니다.

### <kbd>method</kbd> `Team.__init__`

```python
__init__(client, name, attrs=None)
```








---

### <kbd>classmethod</kbd> `Team.create`

```python
create(api, team, admin_username=None)
```

새로운 팀을 생성합니다.



**ARG:**
 
 - `api`:  (`Api`) 사용할 api 인스턴스
 - `team`:  (str) 팀 이름
 - `admin_username`:  (str) 팀의 관리자 사용자 이름 (옵션, 기본값: 현재 사용자)



**반환값:**
 `Team` 오브젝트

---

### <kbd>method</kbd> `Team.create_service_account`

```python
create_service_account(description)
```

이 팀을 위한 서비스 계정을 생성합니다.



**ARG:**
 
 - `description`:  (str) 서비스 계정에 대한 설명



**반환값:**
 생성된 서비스 계정의 `Member` 오브젝트, 실패 시 None

---

### <kbd>method</kbd> `Team.invite`

```python
invite(username_or_email, admin=False)
```

사용자를 팀에 초대합니다.



**ARG:**
 
 - `username_or_email`:  (str) 초대하고자 하는 사용자의 사용자 이름 또는 이메일 어드레스
 - `admin`:  (bool) 해당 사용자를 팀 관리자 권한으로 초대할지 여부, 기본값은 `False`



**반환값:**
 성공 시 `True`, 이미 초대되었거나 사용자가 존재하지 않을 경우 `False`

---