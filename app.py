import streamlit as st
import pandas as pd
import joblib

# Carregar modelo
@st.cache_resource
def load_model():
    return joblib.load("modelo_queimadas_riscofogo_rf.pkl")

model = load_model()

st.title("🔥 Previsão de Risco de Fogo no Brasil (2011–2021)")

st.markdown("""
Este aplicativo utiliza um modelo de Machine Learning treinado em dados reais de queimadas
para prever o **risco de fogo (`riscofogo`)** com base em características da região.
""")

# Entradas do usuário
col1, col2 = st.columns(2)

with col1:
    ano = st.number_input("Ano", min_value=2011, max_value=2035, value=2021)
    mes = st.number_input("Mês", min_value=1, max_value=12, value=8)
    estado = st.text_input("Estado (ex.: SAO PAULO)", "SAO PAULO")
    municipio = st.text_input("Município", "ITAPEVA")
    bioma = st.text_input("Bioma", "Mata Atlantica")

with col2:
    satelite = st.text_input("Satélite", "AQUA_M-T")
    diasemchuva = st.number_input("Dias sem chuva", min_value=0, value=10)
    precipitacao = st.number_input("Precipitação (mm)", min_value=0.0, value=0.0)
    latitude = st.number_input("Latitude", value=-23.0)
    longitude = st.number_input("Longitude", value=-47.0)
    frp = st.number_input("FRP", min_value=0.0, value=10.0)

# Montar DataFrame
input_dict = {
    "satelite": [satelite],
    "estado": [estado],
    "municipio": [municipio],
    "bioma": [bioma],
    "diasemchuva": [diasemchuva],
    "precipitacao": [precipitacao],
    "latitude": [latitude],
    "longitude": [longitude],
    "frp": [frp],
    "ano": [ano],
    "mes": [mes],
}

X_new = pd.DataFrame(input_dict)

# Previsão
if st.button("🔍 Prever risco de fogo"):
    try:
        pred = model.predict(X_new)[0]
        st.success(f"🔥 Risco de fogo previsto: **{pred:.2f}**")
    except Exception as e:
        st.error("Erro ao gerar previsão. Verifique os valores inseridos.")
        st.error(str(e))
