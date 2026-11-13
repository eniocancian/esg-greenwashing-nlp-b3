import os
import re
import glob
import numpy as np
import pandas as pd

# ==============================================================================
# CONFIGURAÇÕES DE DIRETÓRIO (RAIZ DO PROJETO)
# ==============================================================================
PROJECT_ROOT = r"C:\Users\sueli\Downloads\Projetos\esg-greenwashing-nlp-b3"
INTERIM_DIR = os.path.join(PROJECT_ROOT, "data", "interim")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ==============================================================================
# LÉXICOS BILÍNGUES (PT / EN) PARA DETECÇÃO DE GREENWASHING E ESPECIFICIDADE
# ==============================================================================
CLIMATE_TERMS = {
    "pt": [
        r"\bclima\w*\b", r"\bclimátic\w*\b", r"\bcarbono\b", r"\bdescarboniza\w*\b",
        r"\bemiss\w*\b", r"\bgee\b", r"\befeito estufa\b", r"\bnet[\s-]?zero\b",
        r"\btransição energética\b", r"\benergias? renováve\w*\b", r"\bbiocombustíve\w*\b",
        r"\betanol\b", r"\bbiometano\b", r"\bhidrogênio verde\b", r"\bmudanças? climátic\w*\b",
        r"\bacordo de paris\b", r"\btcfd\b", r"\bescopo [123]\b", r"\bdesmatamento\b",
        r"\bbiodiversidade\b", r"\brestauração ecológica\b", r"\bsequestro de carbono\b"
    ],
    "en": [
        r"\bclimate\b", r"\bclimatic\b", r"\bcarbon\b", r"\bdecarboni[sz]\w*\b",
        r"\bemission\w*\b", r"\bghg\b", r"\bgreenhouse gas\w*\b", r"\bnet[\s-]?zero\b",
        r"\benergy transition\b", r"\brenewable energ\w*\b", r"\bbiofuel\w*\b",
        r"\bethanol\b", r"\bbiomethane\b", r"\bgreen hydrogen\b", r"\bclimate change\b",
        r"\bparis agreement\b", r"\btcfd\b", r"\bscope [123]\b", r"\bdeforestation\b",
        r"\bbiodiversity\b", r"\becological restoration\b", r"\bcarbon sequestration\b"
    ]
}

VAGUE_RHETORIC = {
    "pt": [
        r"\bsustentáve\w*\b", r"\bsustentabilidade\b", r"\bcompromiss\w*\b",
        r"\bcomprometid\w*\b", r"\bbuscamos\b", r"\balinhad\w*\b", r"\bexcelência\b",
        r"\bmelhores práticas\b", r"\bconstrução de um mundo\b", r"\bfuturo mais\b",
        r"\blíder na transição\b", r"\balavanca de valor\b", r"\bvisão de longo prazo\b",
        r"\besforço contínuo\b", r"\bimpacto positivo\b", r"\bresponsabilidade socioambiental\b",
        r"\bdesenvolvimento sustentável\b", r"\btransformar\b", r"\binovação sustentável\b"
    ],
    "en": [
        r"\bsustainable\b", r"\bsustainability\b", r"\bcommitment\w*\b",
        r"\bcommitted\b", r"\bwe seek\b", r"\bwe strive\b", r"\baligned\b",
        r"\bexcellence\b", r"\bbest practices\b", r"\bbuilding a better\b",
        r"\bgreener future\b", r"\bleader in the transition\b", r"\blong-term vision\b",
        r"\bcontinuous effort\b", r"\bpositive impact\b", r"\bsustainable development\b",
        r"\btransforming\b", r"\bsustainable innovation\b"
    ]
}

QUANT_ANCHORS = [
    r"\b\d+(?:[.,]\d+)?\s*%",                          # Percentuais (ex: 35%, 12,5%)
    r"\b\d+(?:[.,]\d+)?\s*(?:tco2e?|mtco2e?|ktco2e?)\b",# Emissões (tCO2e, MtCO2e)
    r"\b\d+(?:[.,]\d+)?\s*(?:mw|gw|mwh|gwh|twh|gj|tj)\b",# Energia e potência
    r"\b\d+(?:[.,]\d+)?\s*(?:milhões|bilhões|million|billion|brl|usd|r\$|\$)\b", # Financeiro
    r"\b\d+(?:[.,]\d+)?\s*(?:hectares|ha|km²|m³|litros|toneladas|tons)\b", # Físico/Ambiental
    r"\b20[2345]\d\b",                                 # Metas temporais explícitas (2025, 2030, 2050)
    r"\b(?:gri|sasb|cdp|sbti|iso\s*14001)\b"           # Padrões auditáveis
]

# ==============================================================================
# FUNÇÕES DE SEGMENTAÇÃO, LIMPEZA DE BOILERPLATE E PONTUAÇÃO
# ==============================================================================
def split_into_sentences(text: str) -> list:
    if not isinstance(text, str) or not text.strip():
        return []
    # Limpa quebras de linha excessivas preservando separação de sentenças
    clean = re.sub(r"\s+", " ", text).strip()
    # Quebra após ponto final, exclamação ou interrogação seguidos de espaço e letra maiúscula
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ])", clean)
    return [s.strip() for s in sentences if len(s.split()) >= 6]


def remove_pdf_boilerplate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove cabeçalhos, rodapés e barras de navegação repetidas dentro de um mesmo PDF.
    """
    df = df.copy()
    df["_norm_text"] = (
        df["sentence"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )

    # Conta em quantas páginas distintas cada sentença aparece dentro do mesmo arquivo
    page_counts = (
        df.groupby(["source_file", "_norm_text"])["page_number"]
        .transform("nunique")
    )

    # Padrões explícitos de barras de navegação e índices de página
    nav_pattern = r"^(sumário|contents|apresentação a axia energia|índice remissivo|relatório integrado \d{2}['’]\d{2})\b"
    is_nav = df["_norm_text"].str.contains(nav_pattern, regex=True)

    df_clean = df[(page_counts <= 3) & (~is_nav)].drop(columns=["_norm_text"]).reset_index(drop=True)
    removed = len(df) - len(df_clean)
    print(f"[BOILERPLATE FILTER] Removidas {removed} sentenças repetidas de cabeçalho/navegação.")
    return df_clean


def count_patterns(text: str, patterns: list) -> int:
    text_lower = text.lower()
    return sum(len(re.findall(pat, text_lower)) for pat in patterns)


def score_sentences(df_sent: pd.DataFrame) -> pd.DataFrame:
    climate_hits = []
    quant_hits = []
    vague_hits = []
    spec_scores = []
    gw_risks = []

    for _, row in df_sent.iterrows():
        sent = row["sentence"]
        lang = row["language"] if row["language"] in ("pt", "en") else "pt"
        wc = max(row["word_count"], 1)

        c_hits = count_patterns(sent, CLIMATE_TERMS[lang])
        q_hits = count_patterns(sent, QUANT_ANCHORS)
        v_hits = count_patterns(sent, VAGUE_RHETORIC[lang])

        # Especificidade: densidade de âncoras quantitativas/auditáveis normalizada
        spec = min(1.0, (q_hits * 3.0) / np.sqrt(wc))

        # Risco de Greenwashing: retórica vaga penalizada inversamente pela especificidade
        raw_vague = (v_hits * 2.0) / np.sqrt(wc)
        gw_risk = min(1.0, raw_vague * (1.0 - spec))

        climate_hits.append(c_hits)
        quant_hits.append(q_hits)
        vague_hits.append(v_hits)
        spec_scores.append(round(spec, 4))
        gw_risks.append(round(gw_risk, 4))

    df_sent["climate_term_hits"] = climate_hits
    df_sent["is_climate_related"] = df_sent["climate_term_hits"] > 0
    df_sent["quant_anchor_hits"] = quant_hits
    df_sent["vague_rhetoric_hits"] = vague_hits
    df_sent["specificity_score"] = spec_scores
    df_sent["greenwashing_risk_index"] = gw_risks
    return df_sent


def load_extracted_pages() -> pd.DataFrame:
    # Procura a saída gerada pelo script 01 em data/interim ou data/processed
    candidates = (
        glob.glob(os.path.join(INTERIM_DIR, "*.parquet"))
        + glob.glob(os.path.join(INTERIM_DIR, "*.csv"))
        + glob.glob(os.path.join(PROCESSED_DIR, "*pages*.parquet"))
        + glob.glob(os.path.join(PROCESSED_DIR, "*pages*.csv"))
    )
    if not candidates:
        raise FileNotFoundError(
            "Nenhum arquivo de páginas extraídas encontrado em data/interim/ ou data/processed/. "
            "Execute o 01_pdf_extractor.py primeiro."
        )
    path = candidates[0]
    print(f"Carregando páginas extraídas de: {path}")
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================
def main():
    df_pages = load_extracted_pages()

    # Identifica automaticamente a coluna de texto extraído da página
    text_col = next((c for c in ["text", "page_text", "content", "texto"] if c in df_pages.columns), None)
    if text_col is None:
        raise KeyError(f"Coluna de texto não encontrada. Colunas disponíveis: {df_pages.columns.tolist()}")

    records = []
    for _, row in df_pages.iterrows():
        sents = split_into_sentences(row[text_col])
        for idx, s in enumerate(sents, start=1):
            records.append({
                "company": row["company"],
                "language": row["language"],
                "year": row.get("year", np.nan),
                "source_file": row["source_file"],
                "page_number": row["page_number"],
                "sentence_id": f"{row['company']}_{row['language']}_p{row['page_number']}_s{idx}",
                "sentence": s,
                "word_count": len(s.split()),
            })

    df_sent = pd.DataFrame(records)
    print(f"Total de sentenças brutas segmentadas: {len(df_sent)}")

    # Aplica limpeza de cabeçalhos e barras de navegação repetidas
    df_sent = remove_pdf_boilerplate(df_sent)

    # Calcula métricas NLP de clima, especificidade e risco de greenwashing
    df_scored = score_sentences(df_sent)

    # Salva base completa pontuada
    scored_path = os.path.join(PROCESSED_DIR, "esg_sentences_scored.parquet")
    df_scored.to_parquet(scored_path, index=False)

    # Salva Top Flags de Greenwashing (risco >= 0.35 e especificidade == 0)
    top_flags = (
        df_scored[df_scored["greenwashing_risk_index"] >= 0.35]
        .sort_values(by=["greenwashing_risk_index", "vague_rhetoric_hits"], ascending=[False, False])
    )
    flags_path = os.path.join(PROCESSED_DIR, "top_greenwashing_flags.csv")
    top_flags.to_csv(flags_path, index=False, encoding="utf-8-sig")

    # Gera tabela consolidada de auditoria por empresa e idioma
    summary = (
        df_scored.groupby(["company", "language"])
        .apply(
            lambda g: pd.Series({
                "total_sentencas": len(g),
                "pct_clima": round(g["is_climate_related"].mean() * 100, 2),
                "especificidade_media": round(g["specificity_score"].mean(), 4),
                "risco_gw_medio": round(g["greenwashing_risk_index"].mean(), 4),
                "sentencas_alto_risco_gw": int((g["greenwashing_risk_index"] >= 0.35).sum()),
                "especificidade_climatica": round(
                    g.loc[g["is_climate_related"], "specificity_score"].mean()
                    if g["is_climate_related"].any()
                    else 0.0,
                    4,
                ),
            }),
            include_groups=False,
        )
        .reset_index()
    )

    summary_path = os.path.join(PROCESSED_DIR, "company_esg_audit_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n=== AUDITORIA COMPARATIVA ESG (POR EMPRESA E IDIOMA) ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()