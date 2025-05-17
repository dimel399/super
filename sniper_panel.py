import streamlit as st
import pandas as pd
import requests
import time
import hashlib
import hmac

# CONFIGURAÇÃO DO APP
st.set_page_config(page_title="Radar de Volume Pro", layout="centered")
st.markdown("<h1 style='text-align: center;'>🚀 Radar de Volume Pro 2.0 (Versão Estável)</h1>", unsafe_allow_html=True)
st.write("Buscando informações privadas da conta...")

# 🔑 INSIRA SUAS CHAVES AQUI
API_KEY = st.secrets["fKgCRTDFr9xyDHCWkSTx1DIJITjTVR8IgAh3UJvWvZfkeHbRdCbts3uW64z3tELB"]  # ou defina diretamente como string: "sua_api_key"
SECRET_KEY = st.secrets["fKgCRTDFr9xyDHCWkSTx1DIJITjTVR8IgAh3UJvWvZfkeHbRdCbts3uW64z3tELB"]  # ou "sua_secret_key"

# 🕒 GERA TIMESTAMP
timestamp = int(time.time() * 1000)

# 🔒 GERA ASSINATURA HMAC
def create_signature(query_string, secret_key):
    return hmac.new(secret_key.encode(), query_string.encode(), hashlib.sha256).hexdigest()

# 📄 CONSULTA À ROTA PRIVADA DA BINANCE
try:
    query_string = f"timestamp={timestamp}"
    signature = create_signature(query_string, SECRET_KEY)
    
    headers = {
        'X-MBX-APIKEY': API_KEY
    }
    
    url = f"https://api.binance.com/api/v3/account?{query_string}&signature={signature}"
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        account_data = response.json()
        balances = account_data['balances']
        df = pd.DataFrame(balances)
        df = df[df['free'] != '0.00000000']  # Mostra apenas moedas com saldo
        st.success("Dados da conta carregados com sucesso!")
        st.dataframe(df)
    else:
        st.error(f"Erro ao conectar à API da Binance: {response.status_code}")
        st.write("Resposta:", response.text)

except Exception as e:
    st.error(f"Erro inesperado: {e}")
