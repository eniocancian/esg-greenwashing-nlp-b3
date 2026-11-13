import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

# ==============================================================================
# CONFIGURAÇÕES DE DIRETÓRIO
# ==============================================================================
PROJECT_ROOT = r"C:\Users\sueli\Downloads\Projetos\esg-greenwashing-nlp-b3"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# ==============================================================================
# STOPWORDS E NORMALIZAÇÃO SEMÂNTICA BILÍNGUE (PT <-> EN)
# ==============================================================================
STOPWORDS_BILINGUAL = [
    # Português
    "de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas", "por", "para",
    "com", "sem", "sob", "sobre", "entre", "até", "após", "como", "mais", "menos",
    "muito", "pouco", "que", "qual", "quais", "quem", "onde", "quando", "porque",
    "pois", "mas", "porém", "contudo", "todavia", "entretanto", "portanto", "logo",
    "assim", "também", "ainda", "já", "não", "sim", "ser", "estar", "ter", "haver",
    "fazer", "ir", "vir", "poder", "dever", "são", "foi", "foram", "será", "serão",
    "tem", "têm", "tinha", "sua", "seu", "suas", "seus", "nosso", "nossa", "nossos",
    "nossas", "este", "esta", "estes", "estas", "esse", "essa", "esses", "essas",
    "aquele", "aquela", "isso", "isto", "aquilo", "um", "uma", "uns", "umas", "ao",
    "aos", "à", "às", "pelo", "pela", "pelos", "pelas", "num", "numa", "ou", "e",
    "os", "as", "me", "te", "se", "lhe", "nos", "vos", "lhes", "relatório", "ano",
    "empresa", "companhia", "forma", "meio", "através", "partir", "bem", "cada",
    # Inglês
    "the", "of", "and", "to", "in", "a", "is", "that", "for", "it", "as", "was",
    "with", "be", "by", "on", "not", "he", "i", "this", "are", "or", "his", "from",
    "at", "which", "but", "have", "an", "had", "they", "you", "were", "their", "one",
    "all", "we", "can", "her", "has", "there", "been", "if", "more", "when", "will",
    "would", "who", "so", "no", "out", "up", "into", "about", "than", "its", "only",
    "other", "new", "some", "could", "time", "these", "two", "may", "then", "do",
    "first", "any", "my", "now", "such", "like", "our", "over", "man", "me", "even",
    "most", "made", "after", "also", "did", "many", "before", "must", "through",
    "back", "years", "where", "much", "your", "way", "well", "down", "should",
    "because", "each", "just", "those", "people", "how", "too", "little", "state",
    "good", "very", "make", "world", "still", "own", "see", "men", "work", "long",
    "get", "here", "between", "both", "life", "being", "under", "never", "day",
    "same", "another", "know", "while", "last", "might", "us", "great", "old",
    "year", "off", "come", "since", "against", "go", "came", "right", "used",
    "report", "company", "including", "based", "within", "across", "per", "total"
]

# Mapeamento de conceitos-chave PT -> EN para alinhamento cross-lingual de tópicos
CONCEPT_BRIDGE_PT_TO_EN = {
    r"\bemiss[õo]es?\b": "emissions",
    r"\bcarbono\b": "carbon",
    r"\bdescarboniza\w*\b": "decarbonization",
    r"\bclimátic\w*\b": "climate",
    r"\bclima\b": "climate",
    r"\benerg\w*\b": "energy",
    r"\brenováve\w*\b": "renewable",
    r"\bbiocombustíve\w*\b": "biofuels",
    r"\betanol\b": "ethanol",
    r"\bágua\b": "water",
    r"\bhídric\w*\b": "water",
    r"\befluentes?\b": "effluents",
    r"\bresíduos?\b": "waste",
    r"\bbiodiversidade\b": "biodiversity",
    r"\bflorest\w*\b": "forest",
    r"\bconservação\b": "conservation",
    r"\brestauração\b": "restoration",
    r"\bgovernança\b": "governance",
    r"\bconselho\b": "board",
    r"\bética\b": "ethics",
    r"\bintegridade\b": "integrity",
    r"\bcorrupção\b": "corruption",
    r"\briscos?\b": "risks",
    r"\bsegurança\b": "safety",
    r"\bsaúde\b": "health",
    r"\btrabalhadores?\b": "workers",
    r"\bempregados?\b": "employees",
    r"\bdiversidade\b": "diversity",
    r"\binclusão\b": "inclusion",
    r"\bmulheres\b": "women",
    r"\bdireitos humanos\b": "human_rights",
    r"\bcomunidades?\b": "communities",
    r"\bfornecedores?\b": "suppliers",
    r"\binvestimentos?\b": "investments",
    r"\bfinanceir\w*\b": "financial",
    r"\breceita\b": "revenue",
    r"\bsustentabilidade\b": "sustainability",
    r"\bsustentáve\w*\b": "sustainable",
    r"\bmetas?\b": "targets",
    r"\bindicadores?\b": "indicators",
    r"\bauditoria\b": "audit",
    r"\btransição\b": "transition",
    r"\boperações\b": "operations",
    r"\boperacional\b": "operational",
    r"\btecnologia\b": "technology",
    r"\binovação\b": "innovation",
}

TOPIC_LABELS = {
    0: "Governança, Ética e Compliance",
    1: "Clima, Emissões GEE e Transição Energética",
    2: "Segurança do Trabalho, Saúde e Pessoas",
    3: "Biodiversidade, Água e Gestão Ambiental",
    4: "Cadeia de Valor, Comunidades e Direitos Humanos",
    5: "Desempenho Financeiro, Capex e Operações",
}


def normalize_for_crosslingual_nlp(text: str, lang: str) -> str:
    t = str(text).lower()
    if lang == "pt":
        for pat, repl in CONCEPT_BRIDGE_PT_TO_EN.items():
            t = re.sub(pat, repl, t)
    return t


import re


def main():
    scored_path = os.path.join(PROCESSED_DIR, "esg_sentences_scored.parquet")
    if not os.path.exists(scored_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {scored_path}. Rode o script 02 primeiro.")

    df = pd.read_parquet(scored_path)
    print(f"Carregadas {len(df)} sentenças limpas para modelagem de tópicos e deriva semântica.")

    # Cria representação alinhada semanticamente entre PT e EN
    df["aligned_text"] = [
        normalize_for_crosslingual_nlp(s, l) for s, l in zip(df["sentence"], df["language"])
    ]

    # Vetorização TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=3500,
        min_df=5,
        max_df=0.85,
        stop_words=STOPWORDS_BILINGUAL,
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(df["aligned_text"])
    feature_names = np.array(vectorizer.get_feature_names_out())

    # Modelagem de Tópicos com NMF (6 macro-pilares ESG)
    n_topics = 6
    nmf = NMF(n_components=n_topics, random_state=42, max_iter=400, init="nndsvda")
    W = nmf.fit_transform(tfidf_matrix)
    H = nmf.components_

    # Atribui tópico dominante e confiança a cada sentença
    topic_probs = W / (W.sum(axis=1, keepdims=True) + 1e-12)
    df["dominant_topic_id"] = np.argmax(topic_probs, axis=1)
    df["topic_confidence"] = np.round(np.max(topic_probs, axis=1), 4)

    # Identifica dinamicamente os top termos de cada tópico
    topic_descriptions = []
    for idx, comp in enumerate(H):
        top_terms = ", ".join(feature_names[np.argsort(comp)[::-1][:8]])
        topic_descriptions.append({
            "topic_id": idx,
            "top_keywords": top_terms,
        })
    df_topics_meta = pd.DataFrame(topic_descriptions)
    print("\n=== TÓPICOS EXTRAÍDOS (CROSS-LINGUAL NMF) ===")
    print(df_topics_meta.to_string(index=False))

    # Salva base enriquecida com tópicos
    df_out = df.drop(columns=["aligned_text"])
    topics_parquet = os.path.join(PROCESSED_DIR, "esg_sentences_with_topics.parquet")
    df_out.to_parquet(topics_parquet, index=False)

    # Consolida distribuição de tópicos por empresa e idioma
    topic_dist = (
        df_out.groupby(["company", "language", "dominant_topic_id"])
        .agg(
            sentence_count=("sentence_id", "count"),
            mean_specificity=("specificity_score", "mean"),
            mean_gw_risk=("greenwashing_risk_index", "mean"),
        )
        .reset_index()
    )
    totals = topic_dist.groupby(["company", "language"])["sentence_count"].transform("sum")
    topic_dist["topic_share_pct"] = np.round((topic_dist["sentence_count"] / totals) * 100, 2)
    topic_dist = topic_dist.merge(df_topics_meta, left_on="dominant_topic_id", right_on="topic_id").drop(columns=["topic_id"])

    cross_topics_path = os.path.join(PROCESSED_DIR, "cross_language_topics.csv")
    topic_dist.to_csv(cross_topics_path, index=False, encoding="utf-8-sig")

    # ==========================================================================
    # CÁLCULO DE DERIVA SEMÂNTICA BILÍNGUE (PT vs EN POR EMPRESA)
    # ==========================================================================
    drift_records = []
    for emp in sorted(df_out["company"].unique()):
        idx_pt = df_out[(df_out["company"] == emp) & (df_out["language"] == "pt")].index
        idx_en = df_out[(df_out["company"] == emp) & (df_out["language"] == "en")].index

        # Vetor médio TF-IDF (discurso geral)
        vec_pt = np.asarray(tfidf_matrix[idx_pt].mean(axis=0))
        vec_en = np.asarray(tfidf_matrix[idx_en].mean(axis=0))
        lexical_sim = float(cosine_similarity(vec_pt, vec_en)[0, 0])

        # Vetor médio de distribuição temática (ênfase estratégica ESG)
        top_pt = topic_probs[idx_pt].mean(axis=0).reshape(1, -1)
        top_en = topic_probs[idx_en].mean(axis=0).reshape(1, -1)
        thematic_sim = float(cosine_similarity(top_pt, top_en)[0, 0])

        # Deriva Semântica = 1 - Similaridade Temática (quanto maior, maior o desvio PT vs EN)
        semantic_drift = round(1.0 - thematic_sim, 4)

        drift_records.append({
            "company": emp,
            "sentencas_pt": len(idx_pt),
            "sentencas_en": len(idx_en),
            "similaridade_lexical_pt_en": round(lexical_sim, 4),
            "similaridade_tematica_pt_en": round(thematic_sim, 4),
            "deriva_semantica_bilingue": semantic_drift,
        })

    df_drift = pd.DataFrame(drift_records)
    drift_path = os.path.join(PROCESSED_DIR, "bilingual_semantic_drift.csv")
    df_drift.to_csv(drift_path, index=False, encoding="utf-8-sig")

    print("\n=== DERIVA SEMÂNTICA BILÍNGUE (PT vs EN POR EMPRESA) ===")
    print(df_drift.to_string(index=False))


if __name__ == "__main__":
    main()