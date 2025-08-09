---
title: 'int


  정수형 (integer) 데이터 타입을 나타냅니다.'
menu:
  reference:
    identifier: ko-ref-query-panel-int
---

## 체이닝 가능한 연산
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

두 값이 서로 다른지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 첫 번째 값입니다. |
| `rhs` | 비교할 두 번째 값입니다. |

#### 반환값
두 값이 서로 다른지 여부

<h3 id="number-modulo"><code>number-modulo</code></h3>

한 [number](number.md)를 다른 [number](number.md)로 나누고 나머지를 반환합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 나눌 [number](number.md) |
| `rhs` | 나누는 [number](number.md) |

#### 반환값
두 [number](number.md)의 나머지

<h3 id="number-mult"><code>number-mult</code></h3>

두 [number](number.md)를 곱합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 첫 번째 [number](number.md) |
| `rhs` | 두 번째 [number](number.md) |

#### 반환값
두 [number](number.md)의 곱

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

[Number](number.md)를 거듭제곱합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 밑이 되는 [number](number.md) |
| `rhs` | 지수 [number](number.md) |

#### 반환값
밑 [number](number.md)가 n제곱된 값

<h3 id="number-add"><code>number-add</code></h3>

두 [number](number.md)를 더합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 첫 번째 [number](number.md) |
| `rhs` | 두 번째 [number](number.md) |

#### 반환값
두 [number](number.md)의 합

<h3 id="number-sub"><code>number-sub</code></h3>

한 [number](number.md)에서 다른 [number](number.md)를 뺍니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 빼는 대상 [number](number.md) |
| `rhs` | 빼는 [number](number.md) |

#### 반환값
두 [number](number.md)의 차

<h3 id="number-div"><code>number-div</code></h3>

한 [number](number.md)를 다른 [number](number.md)로 나눕니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 나눌 [number](number.md) |
| `rhs` | 나누는 [number](number.md) |

#### 반환값
두 [number](number.md)의 몫

<h3 id="number-less"><code>number-less</code></h3>

한 [number](number.md)가 다른 [number](number.md)보다 작은지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 [number](number.md) |
| `rhs` | 비교 대상 [number](number.md) |

#### 반환값
첫 번째 [number](number.md)가 두 번째 [number](number.md)보다 작은지 여부

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

한 [number](number.md)가 다른 [number](number.md)보다 작거나 같은지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 [number](number.md) |
| `rhs` | 비교 대상 [number](number.md) |

#### 반환값
첫 번째 [number](number.md)가 두 번째 [number](number.md)보다 작거나 같은지 여부

<h3 id="number-equal"><code>number-equal</code></h3>

두 값이 같은지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 첫 번째 값입니다. |
| `rhs` | 비교할 두 번째 값입니다. |

#### 반환값
두 값이 같은지 여부

<h3 id="number-greater"><code>number-greater</code></h3>

한 [number](number.md)가 다른 [number](number.md)보다 큰지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 [number](number.md) |
| `rhs` | 비교 대상 [number](number.md) |

#### 반환값
첫 번째 [number](number.md)가 두 번째 [number](number.md)보다 큰지 여부

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

한 [number](number.md)가 다른 [number](number.md)보다 크거나 같은지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 [number](number.md) |
| `rhs` | 비교 대상 [number](number.md) |

#### 반환값
첫 번째 [number](number.md)가 두 번째 [number](number.md)보다 크거나 같은지 여부

<h3 id="number-negate"><code>number-negate</code></h3>

[Number](number.md)의 부호를 반전합니다.

| 인수 |  |
| :--- | :--- |
| `val` | 부호를 반전할 숫자 |

#### 반환값
[Number](number.md)

<h3 id="number-toString"><code>number-toString</code></h3>

[Number](number.md)를 문자열로 변환합니다.

| 인수 |  |
| :--- | :--- |
| `in` | 변환할 숫자 |

#### 반환값
해당 [number](number.md)의 문자열 표현

<h3 id="number-toTimestamp"><code>number-toTimestamp</code></h3>

[Number](number.md)를 _타임스탬프_ 로 변환합니다. 값이 31,536,000,000 미만이면 초 단위, 31,536,000,000,000 미만이면 밀리초, 31,536,000,000,000,000 미만이면 마이크로초, 31,536,000,000,000,000,000 미만이면 나노초로 변환됩니다.

| 인수 |  |
| :--- | :--- |
| `val` | 타임스탬프로 변환할 값 |

#### 반환값
타임스탬프

<h3 id="number-abs"><code>number-abs</code></h3>

[Number](number.md)의 절댓값을 계산합니다.

| 인수 |  |
| :--- | :--- |
| `n` | [Number](number.md) |

#### 반환값
해당 [number](number.md)의 절댓값


## 리스트 연산
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

두 값이 서로 다른지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 첫 번째 값입니다. |
| `rhs` | 비교할 두 번째 값입니다. |

#### 반환값
두 값이 서로 다른지 여부

<h3 id="number-modulo"><code>number-modulo</code></h3>

한 [number](number.md)를 다른 [number](number.md)로 나누고 나머지를 반환합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 나눌 [number](number.md) |
| `rhs` | 나누는 [number](number.md) |

#### 반환값
두 [number](number.md)의 나머지

<h3 id="number-mult"><code>number-mult</code></h3>

두 [number](number.md)를 곱합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 첫 번째 [number](number.md) |
| `rhs` | 두 번째 [number](number.md) |

#### 반환값
두 [number](number.md)의 곱

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

[Number](number.md)를 거듭제곱합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 밑이 되는 [number](number.md) |
| `rhs` | 지수 [number](number.md) |

#### 반환값
밑 [number](number.md)가 n제곱된 값

<h3 id="number-add"><code>number-add</code></h3>

두 [number](number.md)를 더합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 첫 번째 [number](number.md) |
| `rhs` | 두 번째 [number](number.md) |

#### 반환값
두 [number](number.md)의 합

<h3 id="number-sub"><code>number-sub</code></h3>

한 [number](number.md)에서 다른 [number](number.md)를 뺍니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 빼는 대상 [number](number.md) |
| `rhs` | 빼는 [number](number.md) |

#### 반환값
두 [number](number.md)의 차

<h3 id="number-div"><code>number-div</code></h3>

한 [number](number.md)를 다른 [number](number.md)로 나눕니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 나눌 [number](number.md) |
| `rhs` | 나누는 [number](number.md) |

#### 반환값
두 [number](number.md)의 몫

<h3 id="number-less"><code>number-less</code></h3>

한 [number](number.md)가 다른 [number](number.md)보다 작은지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 [number](number.md) |
| `rhs` | 비교 대상 [number](number.md) |

#### 반환값
첫 번째 [number](number.md)가 두 번째 [number](number.md)보다 작은지 여부

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

한 [number](number.md)가 다른 [number](number.md)보다 작거나 같은지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 [number](number.md) |
| `rhs` | 비교 대상 [number](number.md) |

#### 반환값
첫 번째 [number](number.md)가 두 번째 [number](number.md)보다 작거나 같은지 여부

<h3 id="number-equal"><code>number-equal</code></h3>

두 값이 같은지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 첫 번째 값입니다. |
| `rhs` | 비교할 두 번째 값입니다. |

#### 반환값
두 값이 같은지 여부

<h3 id="number-greater"><code>number-greater</code></h3>

한 [number](number.md)가 다른 [number](number.md)보다 큰지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 [number](number.md) |
| `rhs` | 비교 대상 [number](number.md) |

#### 반환값
첫 번째 [number](number.md)가 두 번째 [number](number.md)보다 큰지 여부

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

한 [number](number.md)가 다른 [number](number.md)보다 크거나 같은지 확인합니다.

| 인수 |  |
| :--- | :--- |
| `lhs` | 비교할 [number](number.md) |
| `rhs` | 비교 대상 [number](number.md) |

#### 반환값
첫 번째 [number](number.md)가 두 번째 [number](number.md)보다 크거나 같은지 여부

<h3 id="number-negate"><code>number-negate</code></h3>

[Number](number.md)의 부호를 반전합니다.

| 인수 |  |
| :--- | :--- |
| `val` | 부호를 반전할 숫자 |

#### 반환값
[Number](number.md)

<h3 id="numbers-argmax"><code>numbers-argmax</code></h3>

가장 큰 [number](number.md)의 인덱스를 찾습니다.

| 인수 |  |
| :--- | :--- |
| `numbers` | 가장 큰 [number](number.md) 인덱스를 찾을 _리스트_ |

#### 반환값
최대 [number](number.md)의 인덱스

<h3 id="numbers-argmin"><code>numbers-argmin</code></h3>

가장 작은 [number](number.md)의 인덱스를 찾습니다.

| 인수 |  |
| :--- | :--- |
| `numbers` | 가장 작은 [number](number.md) 인덱스를 찾을 _리스트_ |

#### 반환값
최소 [number](number.md)의 인덱스

<h3 id="numbers-avg"><code>numbers-avg</code></h3>

[Numbers](number.md)의 평균을 구합니다.

| 인수 |  |
| :--- | :--- |
| `numbers` | 평균을 구할 [number](number.md) _리스트_ |

#### 반환값
[Numbers](number.md)의 평균

<h3 id="numbers-max"><code>numbers-max</code></h3>

최댓값을 구합니다.

| 인수 |  |
| :--- | :--- |
| `numbers` | 최댓값을 찾을 [number](number.md) _리스트_ |

#### 반환값
최대 [number](number.md)

<h3 id="numbers-min"><code>numbers-min</code></h3>

최솟값을 구합니다.

| 인수 |  |
| :--- | :--- |
| `numbers` | 최소값을 찾을 [number](number.md) _리스트_ |

#### 반환값
최소 [number](number.md)

<h3 id="numbers-stddev"><code>numbers-stddev</code></h3>

[Numbers](number.md)의 표준편차를 계산합니다.

| 인수 |  |
| :--- | :--- |
| `numbers` | 표준편차를 계산할 [number](number.md) _리스트_ |

#### 반환값
[Numbers](number.md)의 표준편차

<h3 id="numbers-sum"><code>numbers-sum</code></h3>

[Numbers](number.md)의 합을 구합니다.

| 인수 |  |
| :--- | :--- |
| `numbers` | 합을 구할 [number](number.md) _리스트_ |

#### 반환값
[Numbers](number.md)의 합

<h3 id="number-toString"><code>number-toString</code></h3>

[Number](number.md)를 문자열로 변환합니다.

| 인수 |  |
| :--- | :--- |
| `in` | 변환할 숫자 |

#### 반환값
해당 [number](number.md)의 문자열 표현

<h3 id="number-toTimestamp"><code>number-toTimestamp</code></h3>

[Number](number.md)를 _타임스탬프_ 로 변환합니다. 값이 31,536,000,000 미만이면 초 단위, 31,536,000,000,000 미만이면 밀리초, 31,536,000,000,000,000 미만이면 마이크로초, 31,536,000,000,000,000,000 미만이면 나노초로 변환됩니다.

| 인수 |  |
| :--- | :--- |
| `val` | 타임스탬프로 변환할 값 |

#### 반환값
타임스탬프

<h3 id="number-abs"><code>number-abs</code></h3>

[Number](number.md)의 절댓값을 계산합니다.

| 인수 |  |
| :--- | :--- |
| `n` | [Number](number.md) |

#### 반환값
해당 [number](number.md)의 절댓값