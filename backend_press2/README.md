## 現在までの処理

画面1  
↓  
Flaskで受け取り  
↓  
CSV保存  
↓  
`exchange_rate.py` 呼び出し  
↓  
為替取得  
↓  
画面2に表示  

---

## ファイル構成

- `backend_press2/country_bean.py`  
  バックエンドメイン（Flask）

- `exchange_rate_api/exchange_rate.py`  
  為替取得処理

- `templates/index.html`  
  仮の画面1（入力画面）

- `templates/result.html`  
  仮の画面2（結果表示画面）

---

## 変数名

### フォーム送信（POST）

- 原産地名  
  - `country`

- 豆の種類  
  - `bean`

---

### 画面2表示用（Flask → HTML）

- 為替  
  - `ex_rate`

- 為替取得日時  
  - `today_date`