import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# CONFIGURAÇÕES DE DIRETÓRIO E ESTILO VISUAL
# ==============================================================================
PROJECT_ROOT = r"C:\Users\sueli\Downloads\Projetos\esg-greenwashing-nlp-b3"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8

COMPANY_LABELS = {
    "eletrobras": "Eletrobras",
    "petrobras": "Petrobras",
    "raizen": "Raízen",
    "suzano": "Suzano",
}
LANG_PALETTE = {"pt": "#1f77b4", "en": "#ff7f0e"}


def main():
    summary = pd.read_csv(os.path.join(PROCESSED_DIR, "company_esg_audit_summary.csv"))
    drift = pd.read_csv(os.path.join(PROCESSED_DIR, "bilingual_semantic_drift.csv"))

    summary["company_label"] = summary["company"].map(COMPANY_LABELS)
    drift["company_label"] = drift["company"].map(COMPANY_LABELS)

    # ==========================================================================
    # FIGURA 1: ESPECIFICIDADE TÉCNICA VS. RISCO DE GREENWASHING (PT vs EN)
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(11, 7))

    for lang, marker in [("pt", "o"), ("en", "s")]:
        sub = summary[summary["language"] == lang]
        ax.scatter(
            sub["especificidade_media"],
            sub["risco_gw_medio"],
            s=sub["total_sentencas"] / 12,
            c=LANG_PALETTE[lang],
            marker=marker,
            alpha=0.88,
            edgecolors="black",
            linewidth=1.2,
            label=f"Idioma: {lang.upper()}",
            zorder=4,
        )

    # Conecta os pares PT <-> EN de cada empresa para mostrar o deslocamento bilíngue
    for emp in summary["company"].unique():
        pair = summary[summary["company"] == emp].sort_values("language")
        ax.plot(
            pair["especificidade_media"],
            pair["risco_gw_medio"],
            linestyle="--",
            color="#7f7f7f",
            alpha=0.65,
            linewidth=1.3,
            zorder=3,
        )
        for _, row in pair.iterrows():
            offset_y = 0.0012 if row["language"] == "en" else -0.0018
            ax.annotate(
                f"{row['company_label']} ({row['language'].upper()})",
                (row["especificidade_media"], row["risco_gw_medio"]),
                xytext=(6, 4 if row["language"] == "en" else -12),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                color="#222222",
            )

    ax.axvline(summary["especificidade_media"].mean(), color="#999999", linestyle=":", linewidth=1.2)
    ax.axhline(summary["risco_gw_medio"].mean(), color="#999999", linestyle=":", linewidth=1.2)

    ax.set_title(
        "Auditoria NLP ESG (B3): Especificidade Quantitativa vs. Risco de Greenwashing",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Especificidade Técnica Média (Âncoras Numéricas / Auditáveis)", fontsize=11)
    ax.set_ylabel("Índice Médio de Risco de Greenwashing (Retórica Vaga)", fontsize=11)
    ax.legend(frameon=True, facecolor="white", fontsize=10, loc="lower left")

    fig.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, "fig1_esg_specificity_vs_greenwashing.png")
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)

    # ==========================================================================
    # FIGURA 2: ENQUADRAMENTO CLIMÁTICO (% SENTENÇAS) E ESPECIFICIDADE CLIMÁTICA
    # ==========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.barplot(
        data=summary,
        x="company_label",
        y="pct_clima",
        hue="language",
        palette=LANG_PALETTE,
        edgecolor="black",
        ax=axes[0],
    )
    axes[0].set_title("Participação do Tema Climático (%)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("% de Sentenças Climáticas", fontsize=11)
    axes[0].legend(title="Idioma", fontsize=9, title_fontsize=10)
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.1f%%", fontsize=9, padding=3)

    sns.barplot(
        data=summary,
        x="company_label",
        y="especificidade_climatica",
        hue="language",
        palette=LANG_PALETTE,
        edgecolor="black",
        ax=axes[1],
    )
    axes[1].set_title("Especificidade Técnica nas Sentenças Climáticas", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Score Médio de Especificidade Climática", fontsize=11)
    axes[1].legend(title="Idioma", fontsize=9, title_fontsize=10)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.3f", fontsize=9, padding=3)

    fig.suptitle(
        "Comunicação Climática Bilíngue (PT vs. EN) — Relatórios ESG B3",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "fig2_climate_focus_and_specificity.png")
    fig.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ==========================================================================
    # FIGURA 3: DERIVA SEMÂNTICA BILÍNGUE (PT <-> EN) POR EMPRESA
    # ==========================================================================
    drift_sorted = drift.sort_values("deriva_semantica_bilingue", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.barh(
        drift_sorted["company_label"],
        drift_sorted["deriva_semantica_bilingue"],
        color=["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"],
        edgecolor="black",
        height=0.55,
    )

    ax.set_xlim(0, 1.0)
    ax.set_title(
        "Deriva Semântica Bilíngue (PT vs. EN) — Distância Temática NMF",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Índice de Deriva Semântica (1 - Similaridade de Cosseno Temática)", fontsize=11)
    ax.set_ylabel("")

    for bar, (_, row) in zip(bars, drift_sorted.iterrows()):
        ax.text(
            bar.get_width() + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{row['deriva_semantica_bilingue']:.4f} (Sim. Lexical: {row['similaridade_lexical_pt_en']:.3f})",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, "fig3_bilingual_semantic_drift.png")
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)

    print("Gráficos gerados com sucesso em:", FIGURES_DIR)
    for p in [fig1_path, fig2_path, fig3_path]:
        print(" -", os.path.basename(p))


if __name__ == "__main__":
    main()