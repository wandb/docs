---
title: Reproduce experiments
menu:
  default:
    identifier: ko-guides-models-track-reproduce_experiments
    parent: track
weight: 7
---

팀 멤버가 생성한 실험을 재현하여 그 결과를 검증하고 확인합니다.

실험을 재현하기 전에 다음 사항을 기록해야 합니다.

* run 이 기록된 project 이름
* 재현하려는 run 이름

실험을 재현하는 방법:

1. run 이 기록된 project 로 이동합니다.
2. 왼쪽 사이드바에서 **Workspace** 탭을 선택합니다.
3. run 목록에서 재현하려는 run 을 선택합니다.
4. **Overview** 를 클릭합니다.

계속하려면 특정 해시에서 실험 코드를 다운로드하거나 실험의 전체 저장소를 복제합니다.

{{< tabpane text=true >}}
{{% tab "Python 스크립트 또는 노트북 다운로드" %}}

실험의 Python 스크립트 또는 노트북을 다운로드합니다.

1. **Command** 필드에서 실험을 생성한 스크립트 이름을 기록합니다.
2. 왼쪽 네비게이션 바에서 **Code** 탭을 선택합니다.
3. 스크립트 또는 노트북에 해당하는 파일 옆에 있는 **Download** 를 클릭합니다.

{{% /tab %}}
{{% tab "GitHub" %}}

팀 멤버가 실험을 생성할 때 사용한 GitHub 저장소를 복제합니다. 이를 수행하는 방법:

1. 필요한 경우 팀 멤버가 실험을 생성하는 데 사용한 GitHub 저장소에 대한 엑세스 권한을 얻습니다.
2. GitHub 저장소 URL이 포함된 **Git repository** 필드를 복사합니다.
3. 저장소를 복제합니다.
    ```bash
    git clone https://github.com/your-repo.git && cd your-repo
    ```
4. **Git state** 필드를 복사하여 터미널에 붙여넣습니다. Git 상태는 팀 멤버가 실험을 생성하는 데 사용한 정확한 커밋을 체크아웃하는 Git 코맨드 집합입니다. 다음 코드 조각에 지정된 값을 자신의 값으로 바꿉니다.
    ```bash
    git checkout -b "<run-name>" 0123456789012345678901234567890123456789
    ```

{{% /tab %}}
{{< /tabpane >}}

5. 왼쪽 네비게이션 바에서 **Files** 를 선택합니다.
6. `requirements.txt` 파일을 다운로드하여 작업 디렉토리에 저장합니다. 이 디렉토리에는 복제된 GitHub 저장소 또는 다운로드된 Python 스크립트 또는 노트북이 포함되어야 합니다.
7. (권장) Python 가상 환경을 만듭니다.
8. `requirements.txt` 파일에 지정된 요구 사항을 설치합니다.
    ```bash
    pip install -r requirements.txt
    ```

9. 이제 코드와 종속성이 있으므로 스크립트 또는 노트북을 실행하여 실험을 재현할 수 있습니다. 저장소를 복제한 경우 스크립트 또는 노트북이 있는 디렉토리로 이동해야 할 수 있습니다. 그렇지 않으면 작업 디렉토리에서 스크립트 또는 노트북을 실행할 수 있습니다.

{{< tabpane text=true >}}
{{% tab "Python 노트북" %}}

Python 노트북을 다운로드한 경우 노트북을 다운로드한 디렉토리로 이동하여 터미널에서 다음 코맨드를 실행합니다.
```bash
jupyter notebook
```

{{% /tab %}}
{{% tab "Python 스크립트" %}}

Python 스크립트를 다운로드한 경우 스크립트를 다운로드한 디렉토리로 이동하여 터미널에서 다음 코맨드를 실행합니다. `<>`로 묶인 값을 자신의 값으로 바꿉니다.

```bash
python <your-script-name>.py
```

{{% /tab %}}
{{< /tabpane >}}
