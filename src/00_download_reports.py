import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from playwright.sync_api import sync_playwright

RAW_DIR = os.path.join("data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

MISSING_RAIZEN = [
    ("PT_BR", "2023 | 2024", "raizen_2023_pt.pdf"),
    ("PT_BR", "2025 | 2026", "raizen_2025_pt.pdf"),
    ("EN",    "2024 | 2025", "raizen_2024_en.pdf"),
]

URL_RAIZEN = "https://www.raizen.com.br/sustentabilidade/reportes-e-agenda-externa"

def _worker_download():
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    with sync_playwright() as p:
        # Mude headless=False se quiser acompanhar na tela
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        print(f"Acessando {URL_RAIZEN}...")
        page.goto(URL_RAIZEN, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(3000)

        try:
            page.get_by_role("button", name="Accept all").click(timeout=5000)
        except Exception:
            pass

        for lang, cycle, target_filename in MISSING_RAIZEN:
            target_path = os.path.join(RAW_DIR, target_filename)

            if os.path.exists(target_path):
                print(f"[PULADO] {target_filename} já existe em {RAW_DIR}.")
                continue

            print(f"Buscando ciclo '{cycle}' ({lang}) -> {target_filename}...")

            # Clica no seletor de idioma e aguarda a atualização do DOM
            if lang == "EN":
                page.locator("text=EN").first.click()
            else:
                page.locator("text=PT_BR").first.click()
            
            page.wait_for_load_state("domcontentloaded")
            page.wait_for_timeout(2500)

            try:
                with page.expect_download(timeout=30000) as download_info:
                    card = page.locator(f"div:has-text('{cycle}')").filter(
                        has_text="Relatório Integrado" if lang == "PT_BR" else "Integrated Report"
                    ).last
                    card.locator("text=/Faça o download|Download/i").first.click()

                download = download_info.value
                download.save_as(target_path)
                print(f"[OK] Salvo com sucesso: {target_path}")
            except Exception as e:
                print(f"[ERRO] Falha ao baixar {target_filename}: {e}")

        browser.close()

def download_missing_raizen():
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_worker_download)
        future.result()

if __name__ == "__main__":
    download_missing_raizen()