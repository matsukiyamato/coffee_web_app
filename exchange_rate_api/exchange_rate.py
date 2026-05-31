import os
import requests
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
dotenv_path = os.path.join(ROOT_DIR, '.env')

# パスを指定して環境変数をロード
load_dotenv(dotenv_path)
API_KEY = os.getenv("API_KEY")

def get_exchange_rate():
    # APIキーが読み込めていない場合の警告
    if not API_KEY:
        print("【警告】APIキーが読み込めませんでした。.envファイルを確認してください。")
        return 150.0  # エラー時はアプリを落とさないよう仮のレート(150円)を返す

    url = f"https://v6.exchangerate-api.com/v6/{API_KEY}/latest/USD"
    
    try:
        res = requests.get(url)
        data = res.json()

        # もしAPIからエラーが返ってきた場合の安全対策
        if "conversion_rates" not in data:
            print(f"【APIエラー】為替データの取得に失敗しました: {data}")
            return 150.0  # エラー時は仮のレート(150円)を返す

        usd_jpy = data["conversion_rates"]["JPY"]
        return usd_jpy

    except Exception as e:
        print(f"【通信エラー】為替APIとの通信に失敗しました: {e}")
        return 150.0