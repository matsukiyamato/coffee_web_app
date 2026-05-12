import streamlit as st
import pandas as pd
import yfinance as yf
import pickle
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- 1. ページの設定 ---
st.set_page_config(page_title="Coffee Price Predictor", page_icon="☕", layout="wide")

st.title("☕ コーヒー価格 AI予測ダッシュボード")
st.markdown("""

為替レート（USD/JPY）とブラジルの気象データ（サンパウロの気温・降水量）から、アラビカ種コーヒーの先物価格を予測します。
""")

# --- 2. モデルの読み込み ---
@st.cache_resource
def load_model():
    try:
        with open('coffee_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("エラー: 'coffee_model.pkl' が見つかりません。app.py と同じフォルダに配置してください。")
        st.stop()

model = load_model()

# --- 3. データの取得 ---
@st.cache_data(ttl=3600) # 1時間ごとにデータを更新
def get_latest_data():
    try:
        # A. 市場データ (為替 & コーヒー)
        market_data = yf.download(["JPY=X", "KC=F"], period="30d", interval="1d")
        
        # 欠損値(NaN)を除外して直近のデータを取得
        jpy_series = market_data['Close']['JPY=X'].dropna()
        coffee_series = market_data['Close']['KC=F'].dropna()
        
        latest_jpy = jpy_series.iloc[-1]
        latest_coffee = coffee_series.iloc[-1]
        
        # B. 気象データ (ブラジル・サンパウロ)
        weather_url = "https://api.open-meteo.com/v1/forecast?latitude=-23.55&longitude=-46.63&daily=temperature_2m_mean,precipitation_sum&timezone=America%2FSao_Paulo"
        weather_res = requests.get(weather_url).json()
        
        temp_mean = weather_res['daily']['temperature_2m_mean'][0]
        precip_sum = weather_res['daily']['precipitation_sum'][0]
        temp_lag3 = weather_res['daily']['temperature_2m_mean'][3] # 簡易的なラグ
        precip_lag7 = weather_res['daily']['precipitation_sum'][6] # 簡易的なラグ
        
        return latest_jpy, latest_coffee, coffee_series, temp_mean, precip_sum, temp_lag3, precip_lag7
        
    except Exception as e:
        st.error(f"データ取得中にエラーが発生しました: {e}")
        st.stop()

# --- メイン処理 ---
with st.spinner('最新の市場・気象データを取得し、AIが予測を計算しています...'):
    jpy, coffee_now, history_series, temp, precip, temp3, precip7 = get_latest_data()

    # 特徴量: [Coffee_Price_Lag1, USD_JPY_Lag1, Temp_Mean, Precip_Sum, Temp_Lag3, Precip_Lag7]
    # 本来は前日の値(Lag1)が必要ですが、リアルタイム性を持たせるため直近値を使用
    features = [[coffee_now, jpy, temp, precip, temp3, precip7]]
    
    # 予測の実行
    predicted_price = model.predict(features)[0]

# --- 4. 画面表示（ダッシュボード） ---
col1, col2, col3 = st.columns(3)

col1.info("📊 現在のコーヒー価格")
col1.metric("アラビカ種 (USD/lb)", f"${coffee_now:.2f}")

col2.warning("💱 現在の為替レート")
col2.metric("USD / JPY", f"¥{jpy:.2f}")

col3.success("🤖 AIによる明日の予測")
diff = predicted_price - coffee_now
col3.metric("予測価格 (USD/lb)", f"${predicted_price:.2f}", f"{diff:+.2f} USD")

st.divider()

# --- 5. グラフの描画 ---
st.subheader("📈 過去30日間の価格推移とAIの予測")

fig = go.Figure()

# 過去の実績
fig.add_trace(go.Scatter(
    x=history_series.index, 
    y=history_series.values, 
    mode='lines+markers',
    name="過去の実績価格", 
    line=dict(color='SaddleBrown', width=3)
))


import streamlit as st

# --- (既存のコードの適当な場所に追加) ---

st.header("💡 なぜこのAIは『天気』と『為替』を見ているの？")

st.info('''
**1. なぜ「為替（ドル/円）」のデータを使っているの？** コーヒー豆は、国際市場において「米ドル」で取引されています [cite: 2]。
そのため、日本の輸入業者にとっては、コーヒー豆自体の価格が変わらなくても、「円安」が進むだけで仕入れコストが跳ね上がってしまいます [cite: 2]。AIには「今の円の価値」を教えることで、日本の経済状況に合ったリアルな価格変動を予測させています。

**2. なぜ「ブラジルの天気」が関係あるの？** ブラジルは世界を代表するコーヒーの主要産地です [cite: 18]。
現地で「異常な高温」が続いたり、雨が極端に降らなかったりすると、「今年はコーヒー豆が不作になるかも！」という不安から、価格が上昇する傾向があります [cite: 20]。AIは、現地の気温や降水量のデータから「数日後の市場の反応」を先読みしています [cite: 20]。

**3. このアプリの強み** 単に過去の価格推移を見るだけでなく、「為替（経済の動き）」と「ブラジルの天気（供給側の状況）」という、現実社会の重要な要因を合体させています [cite: 27]。これにより、より実社会に即した「明日の価格予測」を実現しています。
''')



# 予測点（最後の実績点から線を引く）
last_date = history_series.index[-1]
next_date = last_date + pd.Timedelta(days=1)

fig.add_trace(go.Scatter(
    x=[last_date, next_date], 
    y=[history_series.values[-1], predicted_price], 
    mode='lines+markers',
    name="AI予測", 
    line=dict(color='Crimson', width=3, dash='dash'),
    marker=dict(size=10, symbol='star')
))

fig.update_layout(
    xaxis_title="日付",
    yaxis_title="価格 (USD/lb)",
    hovermode="x unified",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

st.plotly_chart(fig, use_container_width=True)

# --- 6. 解説セクション ---
with st.expander("📝 予測モデルの解説（データソースとロジック）"):
    st.write(f"""
    - **ベースモデル**: 重回帰分析 (Linear Regression)
    - **主要な入力データ (現在値)**:
        - ブラジルの平均気温: **{temp}℃**
        - ブラジルの降水量: **{precip}mm**
    - **なぜこれらのデータが必要なのか？**:
        コーヒーの国際価格は米ドル建てで決定されるため、日本の輸入業者にとって**為替（USD/JPY）**の影響は甚大です。また、世界最大の生産国である**ブラジルの気象状況（特に気温と降水量）**は、将来の供給量を示す先行指標となるため、モデルに組み込むことで予測精度（MAE）の向上を実現しています。
    """)

    import streamlit as st

# --- (既存のコードの適当な場所に追加) ---

st.markdown("---")
with st.expander("👨‍💻 開発の裏側：このアプリはどうやって作られた？（採用担当者様・一般の方向け）"):
    st.markdown("""
    このアプリは、単にAIのプログラムを書いただけではなく、**「実際のIT企業の開発現場」**を想定した本格的なアプローチで作られています。

    **1. プロと同じ「本格的な開発環境」を使っています**
    初心者向けの簡易な練習ツールではなく、実際のシステム開発の現場で標準的に使われるシステム環境（Linux）と、プロ御用達の開発ツール（VS Code）を自ら構築して開発を行いました。これにより、「現場に入ってすぐ即戦力として動ける基礎力」を証明しています。

    **2. 「いつ・何を・なぜ修正したか」をすべて記録しています**
    開発の過程は「GitHub（ギットハブ）」というプロ向けの履歴管理ツールで細かく記録を残しています。これは、ただ動くものを作るだけでなく、「チームメンバーと協力しながら、計画的かつ安全に開発プロセスを管理できる」という、エンジニアとしての確かな管理能力の証です。

    **3. データ集めから画面作りまで、すべて一人で完結させました**
    「為替や天気がコーヒー価格に影響する」という実社会のビジネス課題を紐解き、必要なデータを自動で集め、AIに学習させ、皆さんが今見ているこの画面（Webアプリ）として公開するまでの全工程を一人で実装しました。技術を「実社会で使える形」にする実行力を見ていただければ幸いです。
    """)