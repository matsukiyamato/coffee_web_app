import pandas as pd
from flask import Flask, render_template, jsonify
import json
import os

app = Flask(__name__)

def load_data():
    # 1. Process Brazil.csv
    # Skip metadata rows (first 3 rows based on file inspection)
    try:
        brazil_df = pd.read_csv('Brazil.csv', skiprows=3)
        brazil_df['time'] = pd.to_datetime(brazil_df['time'])
        brazil_df['precipitation_sum (mm)'] = pd.to_numeric(brazil_df['precipitation_sum (mm)'], errors='coerce')
        
        # Monthly precipitation sum
        brazil_df['month_key'] = brazil_df['time'].dt.strftime('%Y-%m')
        monthly_precip = brazil_df.groupby('month_key')['precipitation_sum (mm)'].sum().to_dict()
    except Exception as e:
        print(f"Error processing Brazil.csv: {e}")
        monthly_precip = {}

    # 2. Process CafeArábicaSerieHist.xls - Produção.csv
    try:
        # Header starts at row 5 (index 5)
        prod_df = pd.read_csv('CafeArábicaSerieHist.xls - Produção.csv', skiprows=5)
        # Find 'BRASIL' row
        brasil_row = prod_df[prod_df.iloc[:, 0].astype(str).str.contains('BRASIL', na=False, case=False)]
        
        # Get 2024 production (annual)
        annual_2024 = 0
        if '2024.0' in prod_df.columns and not brasil_row.empty:
            val = str(brasil_row['2024.0'].values[0]).replace(',', '')
            annual_2024 = float(val)
        
        # Estimate monthly production (annual/12)
        monthly_prod_est = annual_2024 / 12 if annual_2024 > 0 else 3300 # fallback
    except Exception as e:
        print(f"Error processing Production CSV: {e}")
        monthly_prod_est = 3300

    # 3. Create chart data for the target period (2024-06 to 2024-11)
    target_months = [
        ('2024-06', '06月'), ('2024-07', '07月'), ('2024-08', '08月'),
        ('2024-09', '09月'), ('2024-10', '10月'), ('2024-11', '11月')
    ]
    
    chart_data = []
    for key, label in target_months:
        chart_data.append({
            "label": label,
            "precip": monthly_precip.get(key, 0.0),
            "prod": monthly_prod_est
        })
        
    return chart_data

@app.route('/')
def index():
    data = load_data()
    # Pass data to the template. In the HTML, you would use {{ chart_data|tojson }}
    return render_template('index.html', chart_data=data)

if __name__ == '__main__':
    app.run(debug=True)