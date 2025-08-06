---
title: 'int


  整数型。'
---

## 連結可能な演算（Chainable Ops）
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

2 つの値が等しくないかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する 2 番目の値。 |

#### 戻り値
2 つの値が等しくない場合に `true` となります。

<h3 id="number-modulo"><code>number-modulo</code></h3>

ある [number](number.md) を別の [number](number.md) で割った余りを返します

| 引数 |  |
| :--- | :--- |
| `lhs` | 割る [number](number.md) |
| `rhs` | 割るための [number](number.md) |

#### 戻り値
2 つの [numbers](number.md) の剰余（モジュロ）

<h3 id="number-mult"><code>number-mult</code></h3>

2 つの [numbers](number.md) を掛けます

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の [number](number.md) |
| `rhs` | 2 番目の [number](number.md) |

#### 戻り値
2 つの [numbers](number.md) の積

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

[ number ](number.md) を指定した指数でべき乗します

| 引数 |  |
| :--- | :--- |
| `lhs` | 基数となる [number](number.md) |
| `rhs` | 指数となる [number](number.md) |

#### 戻り値
基数の [numbers](number.md) を n 乗した値

<h3 id="number-add"><code>number-add</code></h3>

2 つの [numbers](number.md) を加算します

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の [number](number.md) |
| `rhs` | 2 番目の [number](number.md) |

#### 戻り値
2 つの [numbers](number.md) の合計（和）

<h3 id="number-sub"><code>number-sub</code></h3>

[ number ](number.md) から別の [number](number.md) を引きます

| 引数 |  |
| :--- | :--- |
| `lhs` | 減算される [number](number.md) |
| `rhs` | 減算する [number](number.md) |

#### 戻り値
2 つの [numbers](number.md) の差

<h3 id="number-div"><code>number-div</code></h3>

[ number ](number.md) を他の [number](number.md) で割ります

| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる [number](number.md) |
| `rhs` | 割る [number](number.md) |

#### 戻り値
2 つの [numbers](number.md) の商

<h3 id="number-less"><code>number-less</code></h3>

[ number ](number.md) が他と比較して小さいかどうかを判定します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較対象の [number](number.md) |
| `rhs` | 比較する [number](number.md) |

#### 戻り値
最初の [number](number.md) が 2 番目より小さいかどうか

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

[ number ](number.md) が他より小さいか、または等しいかを確認します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較対象の [number](number.md) |
| `rhs` | 比較する [number](number.md) |

#### 戻り値
最初の [number](number.md) が 2 番目の値以下であるかどうか

<h3 id="number-equal"><code>number-equal</code></h3>

2 つの値が等しいかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する 2 番目の値。 |

#### 戻り値
2 つの値が等しければ `true` になります。

<h3 id="number-greater"><code>number-greater</code></h3>

[ number ](number.md) が他より大きいかを確認します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較対象の [number](number.md) |
| `rhs` | 比較する [number](number.md) |

#### 戻り値
最初の [number](number.md) が 2 番目より大きいかどうか

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

[ number ](number.md) が他より大きいか、または等しいかを判定します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較対象の [number](number.md) |
| `rhs` | 比較する [number](number.md) |

#### 戻り値
最初の [number](number.md) が 2 番目の値以上であるかどうか

<h3 id="number-negate"><code>number-negate</code></h3>

[ number ](number.md) の符号を反転します

| 引数 |  |
| :--- | :--- |
| `val` | 符号を反転する number |

#### 戻り値
[ number ](number.md) を返します

<h3 id="number-toString"><code>number-toString</code></h3>

[ number ](number.md) を文字列に変換します

| 引数 |  |
| :--- | :--- |
| `in` | 変換したい number |

#### 戻り値
[ number ](number.md) の文字列表現

<h3 id="number-toTimestamp"><code>number-toTimestamp</code></h3>

[ number ](number.md) を _timestamp_ に変換します。値が 31,536,000,000 未満の場合は秒、31,536,000,000,000 未満の場合はミリ秒、31,536,000,000,000,000 未満の場合はマイクロ秒、そして 31,536,000,000,000,000,000 未満の場合はナノ秒として変換されます。

| 引数 |  |
| :--- | :--- |
| `val` | timestamp に変換する number |

#### 戻り値
タイムスタンプ

<h3 id="number-abs"><code>number-abs</code></h3>

[ number ](number.md) の絶対値を計算します

| 引数 |  |
| :--- | :--- |
| `n` | [ number ](number.md) |

#### 戻り値
[ number ](number.md) の絶対値


## リスト演算（List Ops）
<h3 id="number-notEqual"><code>number-notEqual</code></h3>

2 つの値が等しくないかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する 2 番目の値。 |

#### 戻り値
2 つの値が等しくない場合に `true` となります。

<h3 id="number-modulo"><code>number-modulo</code></h3>

ある [number](number.md) を別の [number](number.md) で割った余りを返します

| 引数 |  |
| :--- | :--- |
| `lhs` | 割る [number](number.md) |
| `rhs` | 割るための [number](number.md) |

#### 戻り値
2 つの [numbers](number.md) の剰余（モジュロ）

<h3 id="number-mult"><code>number-mult</code></h3>

2 つの [numbers](number.md) を掛けます

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の [number](number.md) |
| `rhs` | 2 番目の [number](number.md) |

#### 戻り値
2 つの [numbers](number.md) の積

<h3 id="number-powBinary"><code>number-powBinary</code></h3>

[ number ](number.md) を指定した指数でべき乗します

| 引数 |  |
| :--- | :--- |
| `lhs` | 基数となる [number](number.md) |
| `rhs` | 指数となる [number](number.md) |

#### 戻り値
基数の [numbers](number.md) を n 乗した値

<h3 id="number-add"><code>number-add</code></h3>

2 つの [numbers](number.md) を加算します

| 引数 |  |
| :--- | :--- |
| `lhs` | 最初の [number](number.md) |
| `rhs` | 2 番目の [number](number.md) |

#### 戻り値
2 つの [numbers](number.md) の合計（和）

<h3 id="number-sub"><code>number-sub</code></h3>

[ number ](number.md) から別の [number](number.md) を引きます

| 引数 |  |
| :--- | :--- |
| `lhs` | 減算される [number](number.md) |
| `rhs` | 減算する [number](number.md) |

#### 戻り値
2 つの [numbers](number.md) の差

<h3 id="number-div"><code>number-div</code></h3>

[ number ](number.md) を他の [number](number.md) で割ります

| 引数 |  |
| :--- | :--- |
| `lhs` | 割られる [number](number.md) |
| `rhs` | 割る [number](number.md) |

#### 戻り値
2 つの [numbers](number.md) の商

<h3 id="number-less"><code>number-less</code></h3>

[ number ](number.md) が他と比較して小さいかどうかを判定します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較対象の [number](number.md) |
| `rhs` | 比較する [number](number.md) |

#### 戻り値
最初の [number](number.md) が 2 番目より小さいかどうか

<h3 id="number-lessEqual"><code>number-lessEqual</code></h3>

[ number ](number.md) が他より小さいか、または等しいかを確認します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較対象の [number](number.md) |
| `rhs` | 比較する [number](number.md) |

#### 戻り値
最初の [number](number.md) が 2 番目の値以下であるかどうか

<h3 id="number-equal"><code>number-equal</code></h3>

2 つの値が等しいかどうかを判定します。

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較する最初の値。 |
| `rhs` | 比較する 2 番目の値。 |

#### 戻り値
2 つの値が等しければ `true` になります。

<h3 id="number-greater"><code>number-greater</code></h3>

[ number ](number.md) が他より大きいかを確認します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較対象の [number](number.md) |
| `rhs` | 比較する [number](number.md) |

#### 戻り値
最初の [number](number.md) が 2 番目より大きいかどうか

<h3 id="number-greaterEqual"><code>number-greaterEqual</code></h3>

[ number ](number.md) が他より大きいか、または等しいかを判定します

| 引数 |  |
| :--- | :--- |
| `lhs` | 比較対象の [number](number.md) |
| `rhs` | 比較する [number](number.md) |

#### 戻り値
最初の [number](number.md) が 2 番目の値以上であるかどうか

<h3 id="number-negate"><code>number-negate</code></h3>

[ number ](number.md) の符号を反転します

| 引数 |  |
| :--- | :--- |
| `val` | 符号を反転する number |

#### 戻り値
[ number ](number.md) を返します

<h3 id="numbers-argmax"><code>numbers-argmax</code></h3>

最大値の [number](number.md) のインデックスを取得します

| 引数 |  |
| :--- | :--- |
| `numbers` | 最大値を探すための [numbers](number.md) のリスト |

#### 戻り値
最大値の [number](number.md) のインデックス

<h3 id="numbers-argmin"><code>numbers-argmin</code></h3>

最小値の [number](number.md) のインデックスを取得します

| 引数 |  |
| :--- | :--- |
| `numbers` | 最小値を探すための [numbers](number.md) のリスト |

#### 戻り値
最小値の [number](number.md) のインデックス

<h3 id="numbers-avg"><code>numbers-avg</code></h3>

[ numbers ](number.md) の平均値を求めます

| 引数 |  |
| :--- | :--- |
| `numbers` | 平均を計算する [numbers](number.md) のリスト |

#### 戻り値
[ numbers ](number.md) の平均値

<h3 id="numbers-max"><code>numbers-max</code></h3>

最大値を取得します

| 引数 |  |
| :--- | :--- |
| `numbers` | 最大値を探すための [numbers](number.md) のリスト |

#### 戻り値
最大の [number](number.md)

<h3 id="numbers-min"><code>numbers-min</code></h3>

最小値を取得します

| 引数 |  |
| :--- | :--- |
| `numbers` | 最小値を探すための [numbers](number.md) のリスト |

#### 戻り値
最小の [number](number.md)

<h3 id="numbers-stddev"><code>numbers-stddev</code></h3>

[ numbers ](number.md) の標準偏差を計算します

| 引数 |  |
| :--- | :--- |
| `numbers` | 標準偏差を計算する [numbers](number.md) のリスト |

#### 戻り値
[ numbers ](number.md) の標準偏差

<h3 id="numbers-sum"><code>numbers-sum</code></h3>

[ numbers ](number.md) の合計（サム）を計算します

| 引数 |  |
| :--- | :--- |
| `numbers` | 合計を求める [numbers](number.md) のリスト |

#### 戻り値
[ numbers ](number.md) の合計

<h3 id="number-toString"><code>number-toString</code></h3>

[ number ](number.md) を文字列に変換します

| 引数 |  |
| :--- | :--- |
| `in` | 変換したい number |

#### 戻り値
[ number ](number.md) の文字列表現

<h3 id="number-toTimestamp"><code>number-toTimestamp</code></h3>

[ number ](number.md) を _timestamp_ に変換します。値が 31,536,000,000 未満の場合は秒、31,536,000,000,000 未満の場合はミリ秒、31,536,000,000,000,000 未満の場合はマイクロ秒、そして 31,536,000,000,000,000,000 未満の場合はナノ秒として変換されます。

| 引数 |  |
| :--- | :--- |
| `val` | timestamp に変換する number |

#### 戻り値
タイムスタンプ

<h3 id="number-abs"><code>number-abs</code></h3>

[ number ](number.md) の絶対値を計算します

| 引数 |  |
| :--- | :--- |
| `n` | [ number ](number.md) |

#### 戻り値
[ number ](number.md) の絶対値