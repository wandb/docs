---
title: 숫자
menu:
  reference:
    identifier: ko-ref-query-panel-number
---

## 체이너블 연산(Chainable Ops)
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

두 값이 다른지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 첫 번째 값입니다. |
| `rhs` | 비교할 두 번째 값입니다. |

#### 반환 값
두 값이 서로 다른지 여부입니다.

<h3 id="number-modulo"><code>number-modulo</code></h3>

[숫자](number.md)를 다른 숫자로 나눠서 나머지를 반환합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 나눌 [숫자](number.md) |
| `rhs` | 나눌 [숫자](number.md) (나누는 값) |

#### 반환 값
두 [숫자](number.md)의 모듈로(나머지) 값입니다.

<h3 id="number-mult"><code>number-mult</code></h3>

두 [숫자](number.md)를 곱합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 첫 번째 [숫자](number.md) |
| `rhs` | 두 번째 [숫자](number.md) |

#### 반환 값
두 [숫자](number.md)의 곱입니다.

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

[숫자](number.md)를 지수만큼 거듭제곱합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 밑 [숫자](number.md) |
| `rhs` | 지수 [숫자](number.md) |

#### 반환 값
밑 [숫자](number.md)가 n 제곱 된 값입니다.

<h3 id="number-add"><code>number-add</code></h3>

두 [숫자](number.md)를 더합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 첫 번째 [숫자](number.md) |
| `rhs` | 두 번째 [숫자](number.md) |

#### 반환 값
두 [숫자](number.md)의 합입니다.

<h3 id="number-sub"><code>number-sub</code></h3>

한 [숫자](number.md)에서 다른 [숫자](number.md)를 뺍니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 뺄 대상 [숫자](number.md) |
| `rhs` | 빼는 [숫자](number.md) |

#### 반환 값
두 [숫자](number.md)의 차이입니다.

<h3 id="number-div"><code>number-div</code></h3>

[숫자](number.md)를 다른 숫자로 나눕니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 나눌 [숫자](number.md) |
| `rhs` | 나누는 [숫자](number.md) |

#### 반환 값
두 [숫자](number.md)의 몫입니다.

<h3 id="number-less"><code>number-less</code></h3>

[숫자](number.md)가 다른 숫자보다 작은지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 [숫자](number.md) |
| `rhs` | 비교할 대상 [숫자](number.md) |

#### 반환 값
첫 번째 [숫자](number.md)가 두 번째보다 작은지 여부입니다.

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

[숫자](number.md)가 다른 숫자보다 작거나 같은지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 [숫자](number.md) |
| `rhs` | 비교할 대상 [숫자](number.md) |

#### 반환 값
첫 번째 [숫자](number.md)가 두 번째보다 작거나 같은지 여부입니다.

<h3 id="number-equal"><code>number-equal</code></h3>

두 값이 같은지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 첫 번째 값입니다. |
| `rhs` | 비교할 두 번째 값입니다. |

#### 반환 값
두 값이 같은지 여부입니다.

<h3 id="number-greater"><code>number-greater</code></h3>

[숫자](number.md)가 다른 숫자보다 큰지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 [숫자](number.md) |
| `rhs` | 비교할 대상 [숫자](number.md) |

#### 반환 값
첫 번째 [숫자](number.md)가 두 번째보다 큰지 여부입니다.

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

[숫자](number.md)가 다른 숫자보다 크거나 같은지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 [숫자](number.md) |
| `rhs` | 비교할 대상 [숫자](number.md) |

#### 반환 값
첫 번째 [숫자](number.md)가 두 번째보다 크거나 같은지 여부입니다.

<h3 id="number-negate"><code>number-negate</code></h3>

[숫자](number.md)의 부호를 반전합니다.

| 인수 |  |
| :--- | :--- |
| `val` | 부호를 반전할 숫자 |

#### 반환 값
[숫자](number.md) (부호가 반전된 값)

<h3 id="number-toString"><code>number-toString</code></h3>

[숫자](number.md)를 문자열로 변환합니다.

| 인수 |  |
| :--- | :--- |
| `in` | 변환할 숫자 |

#### 반환 값
[숫자](number.md)의 문자열 표현

<h3 id="number-toTimestamp"><code>number-toTimestamp</code></h3>

[숫자](number.md)를 _타임스탬프_로 변환합니다. 값이 31536000000 미만이면 초로, 31536000000000 미만이면 밀리초로, 31536000000000000 미만이면 마이크로초로, 그리고 31536000000000000000 미만이면 나노초로 변환됩니다.

| 인수 |  |
| :--- | :--- |
| `val` | 타임스탬프로 변환할 숫자 |

#### 반환 값
타임스탬프

<h3 id="number-abs"><code>number-abs</code></h3>

[숫자](number.md)의 절대값을 계산합니다.

| 인수 |  |
| :--- | :--- |
| `n` | [숫자](number.md) |

#### 반환 값
[숫자](number.md)의 절대값

## 리스트 연산(List Ops)
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

두 값이 다른지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 첫 번째 값입니다. |
| `rhs` | 비교할 두 번째 값입니다. |

#### 반환 값
두 값이 서로 다른지 여부입니다.

<h3 id="number-modulo"><code>number-modulo</code></h3>

[숫자](number.md)를 다른 숫자로 나눠서 나머지를 반환합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 나눌 [숫자](number.md) |
| `rhs` | 나눌 [숫자](number.md) (나누는 값) |

#### 반환 값
두 [숫자](number.md)의 모듈로(나머지) 값입니다.

<h3 id="number-mult"><code>number-mult</code></h3>

두 [숫자](number.md)를 곱합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 첫 번째 [숫자](number.md) |
| `rhs` | 두 번째 [숫자](number.md) |

#### 반환 값
두 [숫자](number.md)의 곱입니다.

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

[숫자](number.md)를 지수만큼 거듭제곱합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 밑 [숫자](number.md) |
| `rhs` | 지수 [숫자](number.md) |

#### 반환 값
밑 [숫자](number.md)가 n 제곱 된 값입니다.

<h3 id="number-add"><code>number-add</code></h3>

두 [숫자](number.md)를 더합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 첫 번째 [숫자](number.md) |
| `rhs` | 두 번째 [숫자](number.md) |

#### 반환 값
두 [숫자](number.md)의 합입니다.

<h3 id="number-sub"><code>number-sub</code></h3>

한 [숫자](number.md)에서 다른 [숫자](number.md)를 뺍니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 뺄 대상 [숫자](number.md) |
| `rhs` | 빼는 [숫자](number.md) |

#### 반환 값
두 [숫자](number.md)의 차이입니다.

<h3 id="number-div"><code>number-div</code></h3>

[숫자](number.md)를 다른 숫자로 나눕니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 나눌 [숫자](number.md) |
| `rhs` | 나누는 [숫자](number.md) |

#### 반환 값
두 [숫자](number.md)의 몫입니다.

<h3 id="number-less"><code>number-less</code></h3>

[숫자](number.md)가 다른 숫자보다 작은지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 [숫자](number.md) |
| `rhs` | 비교할 대상 [숫자](number.md) |

#### 반환 값
첫 번째 [숫자](number.md)가 두 번째보다 작은지 여부입니다.

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

[숫자](number.md)가 다른 숫자보다 작거나 같은지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 [숫자](number.md) |
| `rhs` | 비교할 대상 [숫자](number.md) |

#### 반환 값
첫 번째 [숫자](number.md)가 두 번째보다 작거나 같은지 여부입니다.

<h3 id="number-equal"><code>number-equal</code></h3>

두 값이 같은지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 첫 번째 값입니다. |
| `rhs` | 비교할 두 번째 값입니다. |

#### 반환 값
두 값이 같은지 여부입니다.

<h3 id="number-greater"><code>number-greater</code></h3>

[숫자](number.md)가 다른 숫자보다 큰지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 [숫자](number.md) |
| `rhs` | 비교할 대상 [숫자](number.md) |

#### 반환 값
첫 번째 [숫자](number.md)가 두 번째보다 큰지 여부입니다.

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

[숫자](number.md)가 다른 숫자보다 크거나 같은지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 [숫자](number.md) |
| `rhs` | 비교할 대상 [숫자](number.md) |

#### 반환 값
첫 번째 [숫자](number.md)가 두 번째보다 크거나 같은지 여부입니다.

<h3 id="number-negate"><code>number-negate</code></h3>

[숫자](number.md)의 부호를 반전합니다.

| 인수 |  |
| :--- | :--- |
| `val` | 부호를 반전할 숫자 |

#### 반환 값
[숫자](number.md) (부호가 반전된 값)

<h3 id="numbers-argmax"><code>numbers-argmax</code></h3>

가장 큰 [숫자](number.md)의 인덱스를 찾습니다.

| 인수 |  |
| :--- | :--- |
| `numbers` | 최대 [숫자](number.md)를 찾을 _리스트_ |

#### 반환 값
최대 [숫자](number.md)의 인덱스

<h3 id="numbers-argmin"><code>numbers-argmin</code></h3>

가장 작은 [숫자](number.md)의 인덱스를 찾습니다.

| 인수 |  |
| :--- | :--- |
| `numbers` | 최소 [숫자](number.md)를 찾을 _리스트_ |

#### 반환 값
최소 [숫자](number.md)의 인덱스

<h3 id="numbers-avg"><code>numbers-avg</code></h3>

[숫자](number.md)의 평균을 계산합니다.

| 인수 |  |
| :--- | :--- |
| `numbers` | 평균을 낼 [숫자](number.md)의 _리스트_ |

#### 반환 값
[숫자](number.md)의 평균값

<h3 id="numbers-max"><code>numbers-max</code></h3>

최대값을 찾습니다.

| 인수 |  |
| :--- | :--- |
| `numbers` | 최대 [숫자](number.md)를 찾을 _리스트_ |

#### 반환 값
최대 [숫자](number.md)

<h3 id="numbers-min"><code>numbers-min</code></h3>

최소값을 찾습니다.

| 인수 |  |
| :--- | :--- |
| `numbers` | 최소 [숫자](number.md)를 찾을 _리스트_ |

#### 반환 값
최소 [숫자](number.md)

<h3 id="numbers-stddev"><code>numbers-stddev</code></h3>

[숫자](number.md)의 표준편차를 계산합니다.

| 인수 |  |
| :--- | :--- |
| `numbers` | 표준편차를 계산할 [숫자](number.md)의 _리스트_ |

#### 반환 값
[숫자](number.md)의 표준편차

<h3 id="numbers-sum"><code>numbers-sum</code></h3>

[숫자](number.md)의 합을 구합니다.

| 인수 |  |
| :--- | :--- |
| `numbers` | 합계를 구할 [숫자](number.md)의 _리스트_ |

#### 반환 값
[숫자](number.md)의 합

<h3 id="number-toString"><code>number-toString</code></h3>

[숫자](number.md)를 문자열로 변환합니다.

| 인수 |  |
| :--- | :--- |
| `in` | 변환할 숫자 |

#### 반환 값
[숫자](number.md)의 문자열 표현

<h3 id="number-toTimestamp"><code>number-toTimestamp</code></h3>

[숫자](number.md)를 _타임스탬프_로 변환합니다. 값이 31536000000 미만이면 초로, 31536000000000 미만이면 밀리초로, 31536000000000000 미만이면 마이크로초로, 그리고 31536000000000000000 미만이면 나노초로 변환됩니다.

| 인수 |  |
| :--- | :--- |
| `val` | 타임스탬프로 변환할 숫자 |

#### 반환 값
타임스탬프

<h3 id="number-abs"><code>number-abs</code></h3>

[숫자](number.md)의 절대값을 계산합니다.

| 인수 |  |
| :--- | :--- |
| `n` | [숫자](number.md) |

#### 반환 값
[숫자](number.md)의 절대값