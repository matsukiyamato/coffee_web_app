#0519B
from dotenv import load_dotenv
import os
import requests

load_dotenv()
API_KEY = os.getenv("API_KEY")

def get_exchange_rate():

    url = f"https://v6.exchangerate-api.com/v6/{API_KEY}/latest/USD"

    res = requests.get(url)
    data = res.json()

    usd_jpy = data["conversion_rates"]["JPY"]

    return usd_jpy