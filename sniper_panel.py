import streamlit as st
import pandas as pd
import requests
import time
import numpy as np
from datetime import datetime

# --- Configuração das Chaves da Binance ---
API_KEY = "fKgCRTDFr9xyDHCWkSTx1DIJITjTVR8IgAh3UJvWvZfkeHbRdCbts3uW64z3tELB"  # Substitua pela sua API Key real
API_SECRET = "fKgCRTDFr9xyDHCWkSTx1DIJITjTVR8IgAh3UJvWvZfkeHbRdCbts3uW64z3tELB"  # Substitua pela sua Secret Key real

headers = {'X-MBX-APIKEY': API_KEY}
st.set_page_config(page_title="Radar PRO", layout="wide")
st.title("🚀 Radar de Volume Pro 2.0 (Versão Estável)")

# --- Funções de Indicadores Técnicos Manuais ---
def calculate_rsi(df, window=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-10)  # Evita divisão por zero
    return 100 - (100 / (1 + rs))

def calculate_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['close'].ewm(span=fast).mean()
    ema_slow = df['close'].ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return macd, signal_line

def calculate_sma(df, window):
    return df['close'].rolling(window=window).mean()

def calculate_ta(df):
    df['RSI'] = calculate_rsi(df)
    df['MACD'], df['Signal'] = calculate_macd(df)
    df['SMA_50'] = calculate_sma(df, 50)
    df['SMA_200'] = calculate_sma(df, 200)
    return df

# --- Funções Principais com Tratamento de Erros ---
@st.cache_data(ttl=15)
def get_historical_data(symbol, interval='4h', limit=100):
    try:
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        data = requests.get(url, headers=headers).json()
        df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df
    except Exception:
        return None

def backtest_signal(symbol):
    df = get_historical_data(symbol)
    if df is None or len(df) < 20:  # Verifica dados suficientes
        return 0.0
    
    df = calculate_ta(df)
    signals = []
    for i in range(1, len(df)):
        if df['volume'].iloc[i] > 1.5 * df['volume'].rolling(20).mean().iloc[i]:
            if df['close'].iloc[i] > df['open'].iloc[i]:
                signals.append(1)  # Sinal de compra
            else:
                signals.append(-1)  # Sinal de venda
    
    if not signals:
        return 0.0
        
    win_rate = np.mean([1 if (df['close'].iloc[i+1] > df['close'].iloc[i] and signal == 1) or 
                       (df['close'].iloc[i+1] < df['close'].iloc[i] and signal == -1) 
                       else 0 for i, signal in enumerate(signals[:-1])])
    return win_rate

def analyze_symbol(symbol):
    try:
        # Dados em tempo real com verificação
        ticker = requests.get(f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}", headers=headers).json()
        
        # Verificação crítica de dados
        if (float(ticker.get('openPrice', 0)) == 0 or 
            float(ticker.get('quoteVolume', 0)) == 0 or 
            float(ticker.get('volume', 0)) == 0):
            return None
            
        price = float(ticker['lastPrice'])
        open_price = float(ticker['openPrice'])
        quote_volume = float(ticker['quoteVolume'])
        
        # Cálculos protegidos
        price_change = ((price - open_price) / (open_price + 1e-10)) * 100  # Evita divisão por zero
        
        # Dados históricos
        df = get_historical_data(symbol)
        if df is None:
            return None
            
        df = calculate_ta(df)
        if len(df) < 20:  # Verifica dados suficientes
            return None
            
        last_candle = df.iloc[-1]
        avg_volume = df['volume'].mean()
        
        # Filtros Avançados com verificações
        is_volume_spike = quote_volume > 2 * df['volume'].rolling(20).mean().iloc[-1] if not df['volume'].empty else False
        is_uptrend = (last_candle['close'] > last_candle['SMA_50'] > last_candle['SMA_200']) if all(k in last_candle for k in ['SMA_50', 'SMA_200']) else False
        is_rsi_ok = (last_candle['RSI'] < 70 if price_change > 0 else last_candle['RSI'] > 30) if 'RSI' in last_candle else False
        
        win_rate = backtest_signal(symbol)
        
        if is_volume_spike and abs(price_change) > 1 and is_rsi_ok:
            return {
                'Ativo': symbol,
                'Preço': price,
                'Δ Preço %': price_change,
                'Δ Volume %': ((quote_volume - avg_volume) / (avg_volume + 1e-10)) * 100,  # Evita divisão por zero
                'Tendência': 'Alta' if is_uptrend else 'Baixa',
                'RSI': last_candle.get('RSI', 0),
                'Win Rate': f"{win_rate*100:.2f}%",
                'Risk/Reward': f"1:{(last_candle['high'] - price) / (price - last_candle['low'] + 1e-10):.2f}" if price_change > 0 
                              else f"1:{(price - last_candle['low']) / (last_candle['high'] - price + 1e-10):.2f}"
            }
        return None
    except Exception:
        return None  # Silencia erros individuais

# --- Interface Streamlit ---
def main():
    st.sidebar.header("⚙️ Filtros")
    min_volume_change = st.sidebar.slider("Variação Mínima de Volume (%)", 10, 200, 30)
    min_price_change = st.sidebar.slider("Variação Mínima de Preço (%)", 1, 20, 2)
    
    while True:
        try:
            symbols_info = requests.get("https://api.binance.com/api/v3/exchangeInfo", headers=headers).json()
            symbols = [s['symbol'] for s in symbols_info.get('symbols', []) 
                      if s.get('quoteAsset') == 'USDT' and s.get('status') == 'TRADING']
        except Exception:
            st.error("Erro ao carregar símbolos da Binance")
            time.sleep(30)
            st.rerun()
        
        resultados = []
        for symbol in symbols[:100]:  # Limita a 100 ativos para performance
            data = analyze_symbol(symbol)
            if data and abs(data['Δ Volume %']) >= min_volume_change and abs(data['Δ Preço %']) >= min_price_change:
                resultados.append(data)
        
        if resultados:
            df = pd.DataFrame(resultados).sort_values('Δ Volume %', ascending=False)
            
            # Estilo Condicional
            def color_negative_red(val):
                if isinstance(val, (int, float)):
                    color = 'red' if val < 0 else 'green'
                    return f'color: {color}'
                return ''
            
            st.dataframe(
                df.style.applymap(color_negative_red, subset=['Δ Preço %', 'Δ Volume %'])
                .format({
                    'Preço': '{:.8f}',
                    'Δ Preço %': '{:.2f}%',
                    'Δ Volume %': '{:.2f}%',
                    'RSI': '{:.2f}'
                }),
                use_container_width=True,
                height=800
            )
            
            # Gráfico de Exemplo
            if len(df) > 0:
                selected = df.iloc[0]['Ativo']
                df_hist = get_historical_data(selected, interval='1h', limit=24)
                if df_hist is not None:
                    st.line_chart(df_hist.set_index('time')['close'])
        else:
            st.warning("Nenhum ativo encontrado com os critérios atuais.")
        
        time.sleep(30)
        st.rerun()

if __name__ == '__main__':
    main()