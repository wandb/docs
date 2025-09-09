---
title: 数値
menu:
  reference:
    identifier: ja-ref-query-panel-number
---

## チェーン可能な演算
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

2 つの値が等しくないかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する 2 番目の値。 |

#### 戻り値
2 つの値が等しくないかどうか

<h3 id="number-modulo"><code>number-modulo</code></h3>

[number](number.md) を別の数で割り、余りを返す

| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる [number](number.md) |
| `rhs` | 割る数の [number](number.md) |

#### 戻り値
2 つの [numbers](number.md) の剰余

<h3 id="number-mult"><code>number-mult</code></h3>

2 つの [numbers](number.md) を掛ける

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の [number](number.md) |
| `rhs` | 2 番目の [number](number.md) |

#### 戻り値
2 つの [numbers](number.md) の積

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

[number](number.md) を指数で累乗する

| 引数 |  |
| :--- | :--- |
| `lhs` | 底となる [number](number.md) |
| `rhs` | 指数の [number](number.md) |

#### 戻り値
底の [numbers](number.md) を n 乗した値

<h3 id="number-add"><code>number-add</code></h3>

2 つの [numbers](number.md) を加算する

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の [number](number.md) |
| `rhs` | 2 番目の [number](number.md) |

#### 戻り値
2 つの [numbers](number.md) の和

<h3 id="number-sub"><code>number-sub</code></h3>

ある [number](number.md) から別の数を減算する

| 引数 |  |
| :--- | :--- |
| `lhs` | 減算される側の [number](number.md) |
| `rhs` | 減算する [number](number.md) |

#### 戻り値
2 つの [numbers](number.md) の差

<h3 id="number-div"><code>number-div</code></h3>

[number](number.md) を別の数で割る

| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる [number](number.md) |
| `rhs` | 割る数の [number](number.md) |

#### 戻り値
2 つの [numbers](number.md) の商

<h3 id="number-less"><code>number-less</code></h3>

[number](number.md) が別の数より小さいかを確認

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### 戻り値
最初の [number](number.md) が 2 番目より小さいかどうか

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

[number](number.md) が別の数以下かを確認

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### 戻り値
最初の [number](number.md) が 2 番目以下かどうか

<h3 id="number-equal"><code>number-equal</code></h3>

2 つの値が等しいかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する 2 番目の値。 |

#### 戻り値
2 つの値が等しいかどうか

<h3 id="number-greater"><code>number-greater</code></h3>

[number](number.md) が別の数より大きいかを確認

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### 戻り値
最初の [number](number.md) が 2 番目より大きいかどうか

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

[number](number.md) が別の数以上かを確認

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### 戻り値
最初の [number](number.md) が 2 番目以上かどうか

<h3 id="number-negate"><code>number-negate</code></h3>

[number](number.md) の符号を反転する

| 引数 |  |
| :--- | :--- |
| `val` | 符号を反転する数 |

#### 戻り値
[number](number.md)

<h3 id="number-toString"><code>number-toString</code></h3>

[number](number.md) を文字列に変換する

| 引数 |  |
| :--- | :--- |
| `in` | 変換する数 |

#### 戻り値
[number](number.md) の文字列表現

<h3 id="number-toTimestamp"><code>number-toTimestamp</code></h3>

[number](number.md) を _タイムスタンプ_ に変換します。31536000000 未満の値は秒に変換され、31536000000000 未満の値はミリ秒に変換され、31536000000000000 未満の値はマイクロ秒に変換され、31536000000000000000 未満の値はナノ秒に変換されます。

| 引数 |  |
| :--- | :--- |
| `val` | タイムスタンプに変換する数 |

#### 戻り値
タイムスタンプ

<h3 id="number-abs"><code>number-abs</code></h3>

[number](number.md) の絶対値を計算する

| 引数 |  |
| :--- | :--- |
| `n` | 1 つの [number](number.md) |

#### 戻り値
[number](number.md) の絶対値


## リスト演算
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

2 つの値が等しくないかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する 2 番目の値。 |

#### 戻り値
2 つの値が等しくないかどうか

<h3 id="number-modulo"><code>number-modulo</code></h3>

[number](number.md) を別の数で割り、余りを返す

| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる [number](number.md) |
| `rhs` | 割る数の [number](number.md) |

#### 戻り値
2 つの [numbers](number.md) の剰余

<h3 id="number-mult"><code>number-mult</code></h3>

2 つの [numbers](number.md) を掛ける

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の [number](number.md) |
| `rhs` | 2 番目の [number](number.md) |

#### 戻り値
2 つの [numbers](number.md) の積

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

[number](number.md) を指数で累乗する

| 引数 |  |
| :--- | :--- |
| `lhs` | 底となる [number](number.md) |
| `rhs` | 指数の [number](number.md) |

#### 戻り値
底の [numbers](number.md) を n 乗した値

<h3 id="number-add"><code>number-add</code></h3>

2 つの [numbers](number.md) を加算する

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の [number](number.md) |
| `rhs` | 2 番目の [number](number.md) |

#### 戻り値
2 つの [numbers](number.md) の和

<h3 id="number-sub"><code>number-sub</code></h3>

ある [number](number.md) から別の数を減算する

| 引数 |  |
| :--- | :--- |
| `lhs` | 減算される側の [number](number.md) |
| `rhs` | 減算する [number](number.md) |

#### 戻り値
2 つの [numbers](number.md) の差

<h3 id="number-div"><code>number-div</code></h3>

[number](number.md) を別の数で割る

| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる [number](number.md) |
| `rhs` | 割る数の [number](number.md) |

#### 戻り値
2 つの [numbers](number.md) の商

<h3 id="number-less"><code>number-less</code></h3>

[number](number.md) が別の数より小さいかを確認

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### 戻り値
最初の [number](number.md) が 2 番目より小さいかどうか

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

[number](number.md) が別の数以下かを確認

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### 戻り値
最初の [number](number.md) が 2 番目以下かどうか

<h3 id="number-equal"><code>number-equal</code></h3>

2 つの値が等しいかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する 2 番目の値。 |

#### 戻り値
2 つの値が等しいかどうか

<h3 id="number-greater"><code>number-greater</code></h3>

[number](number.md) が別の数より大きいかを確認

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### 戻り値
最初の [number](number.md) が 2 番目より大きいかどうか

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

[number](number.md) が別の数以上かを確認

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する [number](number.md) |
| `rhs` | 比較対象の [number](number.md) |

#### 戻り値
最初の [number](number.md) が 2 番目以上かどうか

<h3 id="number-negate"><code>number-negate</code></h3>

[number](number.md) の符号を反転する

| 引数 |  |
| :--- | :--- |
| `val` | 符号を反転する数 |

#### 戻り値
[number](number.md)

<h3 id="numbers-argmax"><code>numbers-argmax</code></h3>

最大の [number](number.md) のインデックスを求める

| 引数 |  |
| :--- | :--- |
| `numbers` | 最大の [number](number.md) のインデックスを求めるための [numbers](number.md) の _リスト_ |

#### 戻り値
最大の [number](number.md) のインデックス

<h3 id="numbers-argmin"><code>numbers-argmin</code></h3>

最小の [number](number.md) のインデックスを求める

| 引数 |  |
| :--- | :--- |
| `numbers` | 最小の [number](number.md) のインデックスを求めるための [numbers](number.md) の _リスト_ |

#### 戻り値
最小の [number](number.md) のインデックス

<h3 id="numbers-avg"><code>numbers-avg</code></h3>

[numbers](number.md) の平均

| 引数 |  |
| :--- | :--- |
| `numbers` | 平均する対象の [numbers](number.md) の _リスト_ |

#### 戻り値
[numbers](number.md) の平均

<h3 id="numbers-max"><code>numbers-max</code></h3>

最大の数

| 引数 |  |
| :--- | :--- |
| `numbers` | 最大の [number](number.md) を求める [numbers](number.md) の _リスト_ |

#### 戻り値
最大の [number](number.md)

<h3 id="numbers-min"><code>numbers-min</code></h3>

最小の数

| 引数 |  |
| :--- | :--- |
| `numbers` | 最小の [number](number.md) を求める [numbers](number.md) の _リスト_ |

#### 戻り値
最小の [number](number.md)

<h3 id="numbers-stddev"><code>numbers-stddev</code></h3>

[numbers](number.md) の標準偏差

| 引数 |  |
| :--- | :--- |
| `numbers` | 標準偏差を計算する [numbers](number.md) の _リスト_ |

#### 戻り値
[numbers](number.md) の標準偏差

<h3 id="numbers-sum"><code>numbers-sum</code></h3>

[numbers](number.md) の合計

| 引数 |  |
| :--- | :--- |
| `numbers` | 合計する [numbers](number.md) の _リスト_ |

#### 戻り値
[numbers](number.md) の合計

<h3 id="number-toString"><code>number-toString</code></h3>

[number](number.md) を文字列に変換する

| 引数 |  |
| :--- | :--- |
| `in` | 変換する数 |

#### 戻り値
[number](number.md) の文字列表現

<h3 id="number-toTimestamp"><code>number-toTimestamp</code></h3>

[number](number.md) を _タイムスタンプ_ に変換します。31536000000 未満の値は秒に変換され、31536000000000 未満の値はミリ秒に変換され、31536000000000000 未満の値はマイクロ秒に変換され、31536000000000000000 未満の値はナノ秒に変換されます。

| 引数 |  |
| :--- | :--- |
| `val` | タイムスタンプに変換する数 |

#### 戻り値
タイムスタンプ

<h3 id="number-abs"><code>number-abs</code></h3>

[number](number.md) の絶対値を計算する

| 引数 |  |
| :--- | :--- |
| `n` | 1 つの [number](number.md) |

#### 戻り値
[number](number.md) の絶対値