import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime

# --- Configuração das Chaves da Binance ---
API_KEY = "fKgCRTDFr9xyDHCWkSTx1DIJITjTVR8IgAh3UJvWvZfkeHbRdCbts3uW64z3tELB"
API_SECRET = "fKgCRTDFr9xyDHCWkSTx1DIJITjTVR8IgAh3UJvWvZfkeHbRdCbts3uW64z3tELB"
headers = {'X-MBX-APIKEY': API_KEY}

st.set_page_config(page_title="Radar PRO", layout="wide")
st.title("🚀 Radar de Volume Pro 2.0 (Versão Estável)")

# --- Funções de Indicadores Técnicos ---
def calculate_rsi(df, window=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_sma(df, window):
    return df['close'].rolling(window=window).mean()

def calculate_ta(df):
    df['RSI'] = calculate_rsi(df)
    df['MACD'], df['Signal'] = calculate_macd(df)
    df['SMA_50'] = calculate_sma(df, 50)
    df['SMA_200'] = calculate_sma(df, 200)
    return df

# --- Coleta de Dados Históricos ---
@st.cache_data(ttl=15)
def get_historical_data(symbol, interval='4h', limit=100):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        data = requests.get(url, headers=headers).json()
        df = pd.DataFrame(data, columns=[
            'time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df
    except Exception:
        return None

# --- Backtest Simples de Sinal ---
def backtest_signal(symbol):
    df = get_historical_data(symbol)
    if df is None or len(df) < 20:
        return 0.0
    df = calculate_ta(df)
    signals = []
    for i in range(1, len(df)):
        if df['volume'].iloc[i] > 1.5 * df['volume'].rolling(20).mean().iloc[i]:
            if df['close'].iloc[i] > df['open'].iloc[i]:
                signals.append(1)
            else:
                signals.append(-1)
    if not signals:
        return 0.0
    win_rate = np.mean([
        1 if (df['close'].iloc[i+1] > df['close'].iloc[i] and signal == 1) or 
             (df['close'].iloc[i+1] < df['close'].iloc[i] and signal == -1)
        else 0 for i, signal in enumerate(signals[:-1])
    ])
    return win_rate

# --- Análise de Ativo ---
def analyze_symbol(symbol, min_volume_change=30, min_price_change=2):
    try:
        ticker = requests.get(f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}", headers=headers).json()
        price = float(ticker.get('lastPrice', 0))
        open_price = float(ticker.get('openPrice', 0))
        quote_volume = float(ticker.get('quoteVolume', 0))
        if open_price == 0 or quote_volume == 0:
            return None
        price_change = ((price - open_price) / (open_price + 1e-10)) * 100
        df = get_historical_data(symbol)
        if df is None or len(df) < 20:
            return None
        df = calculate_ta(df)
        last = df.iloc[-1]
        avg_volume = df['volume'].mean()
        is_volume_spike = quote_volume > (1 + min_volume_change / 100) * avg_volume
        is_uptrend = last['close'] > last['SMA_50'] > last['SMA_200']
        is_rsi_ok = last['RSI'] < 70 if price_change > 0 else last['RSI'] > 30
        win_rate = backtest_signal(symbol)
        if is_volume_spike and abs(price_change) > min_price_change and is_rsi_ok:
            risk_reward = (
                (last['high'] - price) / (price - last['low'] + 1e-10)
                if price_change > 0
                else (price - last['low']) / (last['high'] - price + 1e-10)
            )
            return {
                'Ativo': symbol,
                'Preço': price,
                'Δ Preço %': price_change,
                'Δ Volume %': ((quote_volume - avg_volume) / (avg_volume + 1e-10)) * 100,
                'Tendência': 'Alta' if is_uptrend else 'Baixa',
                'RSI': last.get('RSI', 0),
                'Win Rate': f"{win_rate*100:.2f}%",
                'Risk/Reward': f"1:{risk_reward:.2f}"
            }
        return None
    except Exception:
        return None

# --- Interface Gráfica ---
def main():
    st.sidebar.header("⚙️ Filtros")
    min_volume_change = st.sidebar.slider("Variação Mínima de Volume (%)", 10, 200, 30)
    min_price_change = st.sidebar.slider("Variação Mínima de Preço (%)", 1, 20, 2)
    st.write("Buscando pares de moedas que atendem aos critérios...")
    try:
        exchange_info = requests.get("https://api.binance.com/api/v3/exchangeInfo", headers=headers).json()
        symbols = [s['symbol'] for s in exchange_info['symbols'] if s['symbol'].endswith("USDT")]
        results = []
        for symbol in symbols:
            analysis = analyze_symbol(symbol, min_volume_change, min_price_change)
            if analysis:
                results.append(analysis)
        if results:
            df = pd.DataFrame(results)
            st.dataframe(df.sort_values(by='Δ Volume %', ascending=False), use_container_width=True)
        else:
            st.info("Nenhum ativo encontrado com os critérios definidos.")
    except Exception as e:
        st.error(f"Erro ao buscar informações: {e}")

if __name__ == "__main__":
    main()
