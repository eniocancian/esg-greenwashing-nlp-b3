import re
import logging
from pathlib import Path
import fitz  # PyMuPDF
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
    if "__file__" in globals()
    else Path(r"C:\Users\sueli\Downloads\Projetos\esg-greenwashing-nlp-b3")
)
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
OUTPUT_PARQUET = INTERIM_DIR / "corpus_pages.parquet"


def infer_metadata(filename: str) -> dict:
    """
    Infere empresa, idioma (pt/en) e ano de referência a partir do nome do arquivo PDF.
    """
    stem = Path(filename).stem.lower()
    parts = stem.split("_")
    company = parts[0] if parts else "unknown"

    lang = "en" if ("_en_" in stem or stem.endswith("_en") or "english" in stem or "full_report" in stem or "sustainability_report" in stem) else "pt"

    year_match = re.search(r"(202[2-6])", stem)
    year = int(year_match.group(1)) if year_match else None

    return {
        "company": company,
        "language": lang,
        "year": year,
        "source_file": filename,
    }


def clean_page_text(raw_text: str) -> str:
    """
    Limpa quebras de linha hifenizadas, espaços múltiplos e linhas isoladas de paginação.
    """
    # Une palavras quebradas por hífen no fim da linha
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", raw_text)
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        # Ignora números de página isolados ou linhas muito curtas sem conteúdo semântico
        if not stripped or re.fullmatch(r"\d{1,4}", stripped):
            continue
        lines.append(stripped)

    cleaned = " ".join(lines)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def extract_corpus():
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        logging.warning(f"Nenhum PDF encontrado em {RAW_DIR}.")
        return

    logging.info(f"Iniciando extração de {len(pdf_files)} relatórios PDF...")
    records = []

    for pdf_path in tqdm(pdf_files, desc="Extraindo PDFs"):
        meta = infer_metadata(pdf_path.name)
        try:
            with fitz.open(pdf_path) as doc:
                total_pages = len(doc)
                for page_idx in range(total_pages):
                    raw_text = doc[page_idx].get_text("text")
                    cleaned_text = clean_page_text(raw_text)

                    # Filtra páginas puramente visuais/capas com menos de 80 caracteres
                    if len(cleaned_text) < 80:
                        continue

                    records.append({
                        **meta,
                        "page_number": page_idx + 1,
                        "total_pages": total_pages,
                        "char_count": len(cleaned_text),
                        "word_count": len(cleaned_text.split()),
                        "text": cleaned_text,
                    })
        except Exception as e:
            logging.error(f"Erro ao ler {pdf_path.name}: {e}")

    df = pd.DataFrame(records)
    df.to_parquet(OUTPUT_PARQUET, index=False)

    logging.info(f"Extração concluída! Total de páginas válidas extraídas: {len(df)}")
    logging.info(f"Dataset salvo em: {OUTPUT_PARQUET}")
    print("\nResumo por Empresa e Idioma:")
    print(df.groupby(["company", "language"])["page_number"].count().rename("paginas_extraidas"))


if __name__ == "__main__":
    extract_corpus()