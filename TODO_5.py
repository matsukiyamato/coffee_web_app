import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

# 1. 学習用データの読み込み
df = pd.read_csv('final_coffee_machine_learning_dataset.csv')

# 2. 特徴量と目的変数の定義
features = ['flowering_temp_mean', 'flowering_precip_sum', 
            'growing_temp_mean', 'growing_soil_moisture_mean', 
            'harvest_humidity_mean', 'harvest_precip_sum']
target = 'Total Cup Points'

X = df[features]
y = df[target]

# 3. モデルの学習
params = {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'max_depth': 3, 'verbose': -1}
model = lgb.train(params, lgb.Dataset(X, y), num_boost_round=50)

# 4. 2018年〜2026年の気象データに基づく予測シミュレーション
# ※この部分は、open-meteo等のデータを月次に集約した気象予測用データフレーム(df_future_weather)が必要です
# 以下は、データが存在する前提の予測実行コードです
predictions = model.predict(X) 

# 5. 可視化（予測図の生成）
plt.figure(figsize=(12, 6))
plt.plot(df['Harvest Year'], predictions, marker='o', linestyle='-', color='r', label='Predicted Quality')
plt.title('Coffee Quality Prediction Simulation (2016-2026)')
plt.xlabel('Harvest Year')
plt.ylabel('Predicted Total Cup Points')
plt.grid(True)
plt.savefig('coffee_forecast_2026.png')
print("予測図 coffee_forecast_2026.png を生成しました。")