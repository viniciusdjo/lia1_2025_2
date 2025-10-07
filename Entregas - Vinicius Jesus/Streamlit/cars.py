import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="Viabilidade de Compra de Carro", layout="centered")
st.title("Viabilidade de Compra de Carro")

st.caption(
    "Dataset: Cars93")
DATA_URL = "https://raw.githubusercontent.com/selva86/datasets/master/Cars93.csv"
USD_PER_UNIT = 1000.0  # 1 unidade do CSV = US$ 1.000


@st.cache_data
def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)

    def norm(c: str) -> str:
        c = re.sub(r'[^0-9a-zA-Z]+', '_', c).strip('_')
        return c
    df.columns = [norm(c) for c in df.columns]
    return df


df = load_data(DATA_URL)

target = "Price"  # em MIL dólares
features = [
    "Type", "AirBags", "DriveTrain",
    "MPG_city", "MPG_highway",
    "Horsepower", "EngineSize",
    "Passengers", "Length", "Width"
]

df_model = df[features + [target]].dropna().reset_index(drop=True)

# categóricas x numéricas
cat_cols = ["Type", "AirBags", "DriveTrain"]
num_cols = [c for c in features if c not in cat_cols]

# pipeline
pipe = Pipeline(steps=[
    ("prep", ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )),
    ("model", RandomForestRegressor(n_estimators=400, random_state=42))
])

X = df_model[features]
y_thousands = df_model[target]  # alvo em MIL dólares

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y_thousands, test_size=0.20, random_state=42)
pipe.fit(X_tr, y_tr)
y_pred_thousands = pipe.predict(X_te)

# métricas de qualidade (mostrar em DÓLARES)
mae = mean_absolute_error(
    y_te, y_pred_thousands)               # em MIL dólares
mae_display = mae * USD_PER_UNIT                                 # em dólares
r2 = r2_score(y_te, y_pred_thousands)

st.subheader("Qualidade do modelo (preço justo)")
c1, c2 = st.columns(2)
c1.metric("MAE (erro médio)", f"US$ {mae_display:,.0f}")
c2.metric("R²", f"{r2:.3f}")

st.markdown("---")
with st.expander("Ver 10 primeiras linhas do dataset"):
    st.dataframe(df[features + [target]].head(10), use_container_width=True)

st.markdown("---")
st.subheader("Simulador de viabilidade")

st.markdown("### Visualizações")

tab1, tab2 = st.tabs(
    ["Dispersão por variável (com reta de regressão)", "Predito vs. Real"])

with tab1:
    st.caption(
        "Escolha uma variável numérica para ver a correlação com o preço (em US$) e uma reta de regressão simples.")
    xcol = st.selectbox("Variável no eixo X", options=num_cols, index=num_cols.index(
        "Horsepower") if "Horsepower" in num_cols else 0)

    # dados completos para a dispersão
    x = df_model[xcol].values.astype(float)
    y_usd = (df_model[target].values.astype(float)) * \
        USD_PER_UNIT  # converter para dólares

    # regressão linear simples via polyfit
    # (reta: y = a*x + b)
    a, b = np.polyfit(x, y_usd, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = a * x_line + b

    # R² da regressão simples (univariada)
    y_hat_uni = a * x + b
    ss_res = np.sum((y_usd - y_hat_uni) ** 2)
    ss_tot = np.sum((y_usd - y_usd.mean()) ** 2)
    r2_uni = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    fig, ax = plt.subplots()
    ax.scatter(x, y_usd)
    ax.plot(x_line, y_line)
    ax.set_xlabel(xcol)
    ax.set_ylabel("Preço (US$)")
    ax.set_title(f"Preço vs. {xcol}  —  R² (linear): {r2_uni:.3f}")
    st.pyplot(fig)

with tab2:
    st.caption(
        "Avaliação do modelo treinado: preço **predito** vs. **real** (em US$) no conjunto de teste.")
    y_real_usd = y_te.to_numpy(dtype=float) * USD_PER_UNIT
    y_pred_usd = y_pred_thousands.astype(float) * USD_PER_UNIT

    # linha de 45°
    min_axis = float(min(y_real_usd.min(), y_pred_usd.min()))
    max_axis = float(max(y_real_usd.max(), y_pred_usd.max()))
    line = np.linspace(min_axis, max_axis, 200)

    fig2, ax2 = plt.subplots()
    ax2.scatter(y_real_usd, y_pred_usd)
    ax2.plot(line, line)
    ax2.set_xlabel("Preço real (US$)")
    ax2.set_ylabel("Preço predito (US$)")
    ax2.set_title("Predito vs. Real (conjunto de teste)")
    st.pyplot(fig2)

st.markdown("---")
st.subheader("Simulador de viabilidade")
# valores padrão (medianas/modas) para inputs
defaults_num = X[num_cols].median(numeric_only=True)
# defaults_cat = {c: X[c].mode().iloc[0] for c in cat_cols}  # (não utilizado, pode remover)

colA, colB = st.columns(2)
with colA:
    Type = st.selectbox("Tipo", sorted(X["Type"].dropna().unique()))
    AirBags = st.selectbox("AirBags", sorted(X["AirBags"].dropna().unique()))
    DriveTrain = st.selectbox("Tração", sorted(
        X["DriveTrain"].dropna().unique()))
    Horsepower = st.number_input(
        "Potência (HP)", value=float(defaults_num["Horsepower"]))
    EngineSize = st.number_input("Motor (L)", value=float(
        defaults_num["EngineSize"]), step=0.1)

with colB:
    MPG_city = st.number_input(
        "Consumo cidade (MPG)", value=float(defaults_num["MPG_city"]))
    MPG_highway = st.number_input(
        "Consumo rodovia (MPG)", value=float(defaults_num["MPG_highway"]))
    Passengers = st.number_input("Passageiros", value=float(
        defaults_num["Passengers"]), step=1.0)
    Length = st.number_input(
        "Comprimento (in)", value=float(defaults_num["Length"]))
    Width = st.number_input("Largura (in)", value=float(defaults_num["Width"]))

st.markdown("##### Preço pedido do anúncio (em dólares)")
asking_price_usd = st.number_input(
    "Preço pedido (US$)", min_value=0.0, step=500.0, value=20000.0)

# prever preço justo (modelo em MIL dólares)
row = pd.DataFrame([{
    "Type": Type, "AirBags": AirBags, "DriveTrain": DriveTrain,
    "MPG_city": MPG_city, "MPG_highway": MPG_highway,
    "Horsepower": Horsepower, "EngineSize": EngineSize,
    "Passengers": Passengers, "Length": Length, "Width": Width
}])
fair_price_thousands = float(pipe.predict(row)[0])      # mil dólares
fair_price_usd = fair_price_thousands * USD_PER_UNIT    # dólares

# comparação pedido vs justo (em dólares)
diff_usd = asking_price_usd - fair_price_usd
pct = (diff_usd / fair_price_usd) * 100 if fair_price_usd else 0.0

if pct <= -5:
    verdict = "✅ Bom negócio (abaixo do preço justo)"
    cor = "green"
elif pct < 5:
    verdict = "🟨 Preço justo (negocie garantias/manutenção)"
    cor = "orange"
else:
    verdict = "❌ Caro para o mercado (renegocie ou evite)"
    cor = "red"

st.markdown("#### Resultado")
st.write(f"**Preço justo estimado:** US$ {fair_price_usd:,.0f}")
st.write(f"**Preço pedido:** US$ {asking_price_usd:,.0f}")
st.write(f"**Diferença:** US$ {diff_usd:,.0f}  (**{pct:+.1f}%**)")
st.markdown(
    f"<div style='padding:10px;border-radius:10px;background:{cor};color:white;font-weight:700'>{verdict}</div>", unsafe_allow_html=True)

# custo mensal de combustível (opcional)
if "km_mes" in locals():
    mpg_mix = (uso_urbano/100)*MPG_city + (1-uso_urbano/100)*MPG_highway
    km_por_litro = mpg_mix * 1.60934 / 3.78541
    if km_por_litro > 0:
        litros_mes = km_mes / km_por_litro
        custo_mes = litros_mes * preco_combust
        st.markdown("#### Estimativa de combustível (mensal)")
        st.write(f"Consumo médio estimado: **{km_por_litro:.1f} km/L**")
        st.write(
            f"Uso mensal: **{litros_mes:.1f} L**  →  **US$ {custo_mes:,.0f}/mês**")
