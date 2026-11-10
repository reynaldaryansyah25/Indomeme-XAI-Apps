"""Browser acceptance against the actual local Streamlit service."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import sys
from PIL import Image
from playwright.sync_api import sync_playwright, expect

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8501')
    parser.add_argument('--mode', choices=['success', 'unavailable'], default='success')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / 'docs'
    from research_acceptance import smoke_image
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, channel='msedge')
        page = browser.new_page(viewport={'width': 1440, 'height': 1400})
        page.goto(args.url)
        page.get_by_role('heading', name='Analisis Meme', exact=True).wait_for()
        for technical in ['ICAM', 'Seed 999', 'best_model.pt', 'IndoBERTweet', 'Ambang keputusan']:
            assert technical not in page.locator('body').inner_text(), technical
        page.screenshot(path=str(output / 'desktop.png'), full_page=True)
        assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
        page.set_viewport_size({'width': 390, 'height': 1800})
        page.screenshot(path=str(output / 'mobile.png'), full_page=True)
        assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
        data = BytesIO()
        smoke_image().save(data, format='PNG')
        page.locator('input[type=file]').set_input_files({'name': 'meme.png', 'mimeType': 'image/png', 'buffer': data.getvalue()})
        page.get_by_text('Detail masukan', exact=True).click()
        page.get_by_role('textbox', name='Teks meme (OCR)').fill('SELAMAT PAGI Semoga harimu menyenangkan')
        page.get_by_role('textbox', name='Deskripsi gambar').fill('Ilustrasi matahari kuning di atas langit biru dan tanah hijau.')
        # No selection maps to research's unknown topic, exactly as the CLI probe.
        page.get_by_role('button', name='Analisis Meme').click()
        if args.mode == 'success':
            page.get_by_text('Analisis selesai.', exact=True).wait_for(timeout=180000)
            expected = json.loads((output / 'runtime_check.json').read_text(encoding='utf-8'))['prediction']
            assert page.locator('.prediction').inner_text() == expected['label']
            displayed = float(page.locator('.probability-top b').inner_text().replace('%', '').replace(',', '.'))/100
            assert abs(displayed-expected['hate_probability']) <= 5e-4
            page.locator('.token-heat').first.wait_for()
            for name in ['Cross-Attention', 'Attention Rollout', 'Incongruity']:
                page.get_by_role('tab', name=name, exact=True).click()
                assert page.get_by_role('tabpanel').filter(visible=True).locator('img').count() == 1
            assert 'Penjelasan metode ini tidak tersedia.' not in page.locator('body').inner_text()
            page.get_by_role('tab', name='Integrated Gradients', exact=True).click()
        else:
            page.get_by_text('Analisis belum dapat dijalankan.', exact=False).wait_for(timeout=20000)
        page.screenshot(path=str(output / 'mobile-upload.png'), full_page=True)
        page.set_viewport_size({'width': 1440, 'height': 1400})
        page.screenshot(path=str(output / 'desktop-result.png'), full_page=True)
        replacement = smoke_image()
        replacement.putpixel((0, 0), (255, 0, 0))
        replacement_data = BytesIO()
        replacement.save(replacement_data, format='PNG')
        page.locator('input[type=file]').set_input_files({'name': 'replacement.png', 'mimeType': 'image/png', 'buffer': replacement_data.getvalue()})
        page.locator('.filename').filter(has_text='replacement.png').wait_for()
        expect(page.locator('.prediction')).to_have_text('—')
        page.get_by_role('button', name='Hapus gambar', exact=True).click()
        page.get_by_text('Belum ada gambar', exact=True).wait_for()
        expect(page.locator('.prediction')).to_have_text('—')
        page.locator('input[type=file]').set_input_files({'name': 'broken.png', 'mimeType': 'image/png', 'buffer': b'not-an-image'})
        page.get_by_text('Gambar rusak atau tidak dapat dibaca.', exact=False).wait_for()
        assert page.get_by_role('button', name='Analisis Meme').is_disabled()
        expect(page.locator('.prediction')).to_have_text('—')
        print(f'PASS: desktop, mobile, real browser upload, {args.mode} analysis, replace/reset, corrupt upload')
        browser.close()

if __name__ == '__main__':
    main()
