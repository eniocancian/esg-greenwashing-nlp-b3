from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="B3 ESG & Greenwashing NLP Auditor",
    page_icon="🌿",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

SENTENCES_PATH = (
    PROCESSED_DIR / "esg_sentences_with_topics.parquet"
    if (PROCESSED_DIR / "esg_sentences_with_topics.parquet").exists()
    else PROCESSED_DIR / "esg_sentences_scored.parquet"
)
SUMMARY_PATH = PROCESSED_DIR / "company_esg_audit_summary.csv"
DRIFT_PATH = PROCESSED_DIR / "bilingual_semantic_drift.csv"
TOPICS_PATH = PROCESSED_DIR / "cross_language_topics.csv"


@st.cache_data
def load_data():
    df_sent = pd.read_parquet(SENTENCES_PATH)
    df_sum = pd.read_csv(SUMMARY_PATH) if SUMMARY_PATH.exists() else pd.DataFrame()
    df_drift = pd.read_csv(DRIFT_PATH) if DRIFT_PATH.exists() else pd.DataFrame()
    df_topics = pd.read_csv(TOPICS_PATH) if TOPICS_PATH.exists() else pd.DataFrame()
    return df_sent, df_sum, df_drift, df_topics


st.title("Auditoria ESG & Detecção de Greenwashing (B3)")
st.markdown(
    "Análise quantitativa de especificidade climática, retórica aspiracional sem lastro e alinhamento bilíngue (**PT-BR vs. EN**) em relatórios de sustentabilidade."
)

if not SENTENCES_PATH.exists():
    st.error("Base processada não encontrada. Execute os scripts 01 a 04 antes de abrir o dashboard.")
    st.stop()

df_sent, df_sum, df_drift, df_topics = load_data()

# Sidebar de Filtros
st.sidebar.header("Filtros de Auditoria")
companies = sorted(df_sent["company"].unique())
selected_companies = st.sidebar.multiselect("Empresas", companies, default=companies)

languages = sorted(df_sent["language"].unique())
selected_langs = st.sidebar.multiselect("Idiomas", languages, default=languages)

only_climate = st.sidebar.checkbox("Somente sentenças climáticas", value=False)
min_gw_risk = st.sidebar.slider("Índice mínimo de risco de Greenwashing", 0.0, 1.0, 0.0, 0.05)

filtered = df_sent[
    (df_sent["company"].isin(selected_companies))
    & (df_sent["language"].isin(selected_langs))
    & (df_sent["greenwashing_risk_index"] >= min_gw_risk)
]
if only_climate:
    filtered = filtered[filtered["is_climate_related"]]

# KPIs Principais
col1, col2, col3, col4 = st.columns(4)
col1.metric("Sentenças Filtradas", f"{len(filtered):,}")
col2.metric("Especificidade Média", f"{filtered['specificity_score'].mean():.4f}" if len(filtered) else "0.0000")
col3.metric("Risco Médio Greenwashing", f"{filtered['greenwashing_risk_index'].mean():.4f}" if len(filtered) else "0.0000")
col4.metric("Alertas Alto Risco (>= 0.4)", f"{int((filtered['greenwashing_risk_index'] >= 0.4).sum()):,}")

st.divider()

# Abas Analíticas
tab1, tab2, tab3 = st.tabs(["Visão Executiva & Gráficos", "Auditoria de Sentenças (Flags)", "Alinhamento Bilíngue & Tópicos"])

with tab1:
    if not df_sum.empty:
        st.subheader("Tabela Comparativa por Empresa e Idioma")
        st.dataframe(df_sum, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    fig1 = FIGURES_DIR / "esg_specificity_vs_greenwashing.png"
    fig2 = FIGURES_DIR / "cross_language_topics.png"
    if fig1.exists():
        c1.image(str(fig1), caption="Especificidade Climática vs. Retórica Vaga", use_container_width=True)
    if fig2.exists():
        c2.image(str(fig2), caption="Distribuição Temática Cross-Language (NMF)", use_container_width=True)

with tab2:
    st.subheader("Inspeção Granular de Sentenças")
    cols_to_show = [
        "company", "language", "page_number", "specificity_score",
        "greenwashing_risk_index", "quant_anchor_hits", "vague_rhetoric_hits", "sentence"
    ]
    if "dominant_topic" in filtered.columns:
        cols_to_show.insert(3, "dominant_topic")

    top_view = filtered.sort_values("greenwashing_risk_index", ascending=False)[cols_to_show].head(100)
    st.dataframe(top_view, use_container_width=True, hide_index=True)

with tab3:
    if not df_drift.empty:
        st.subheader("Alinhamento Semântico PT vs. EN (TF-IDF + LSA)")
        st.dataframe(df_drift, use_container_width=True, hide_index=True)
    if not df_topics.empty:
        st.subheader("Proporção Temática por Empresa e Idioma (%)")
        st.dataframe(df_topics, use_container_width=True, hide_index=True)