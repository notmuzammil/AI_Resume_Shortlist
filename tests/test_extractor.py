import io
import zipfile

import pytest
from conftest import make_docx, make_image_pdf, read, sample_cvs

from modules import extractor
from modules.extractor import ExtractionError, extract_document, normalize_text


def test_txt_utf8_utf16_cp1252():
    s = "Ahmed Ali — Python Developer, café"
    assert "café" in extract_document(s.encode("utf-8"), "txt").text
    assert "café" in extract_document(s.encode("utf-16"), "txt").text
    assert "caf" in extract_document(s.encode("cp1252"), "txt").text
    assert "Python" in extract_document(b"\xef\xbb\xbf" + s.encode("utf-8"), "txt").text


def test_normalize_ligatures_math_letters_quotes():
    t = normalize_text("certiﬁcation 𝐒𝐤𝐢𝐥𝐥𝐬 Bachelor’s​ degree x")
    assert "certification" in t and "Skills" in t and "Bachelor's degree x" in t


@pytest.mark.parametrize("data,ftype,msg", [
    (b"", "pdf", "empty"),
    (b"MZ\x90\x00 not a pdf", "pdf", "signature"),
    (b"just text", "docx", "DOCX"),
    (b"\x00\x01\x02binary\x00", "txt", "binary"),
    (b"hello", "exe", "Unsupported"),
    (b"%PDF-1.4 this is not really a pdf", "pdf", "corrupted"),
])
def test_rejects_bad_files_with_safe_message(data, ftype, msg):
    with pytest.raises(ExtractionError) as e:
        extract_document(data, ftype)
    assert msg.lower() in str(e.value).lower()


def test_oversize_file_rejected(monkeypatch):
    monkeypatch.setattr(extractor, "MAX_FILE_BYTES", 1000)
    with pytest.raises(ExtractionError, match="too large"):
        extract_document(b"%PDF-" + b"x" * 2000, "pdf")


def test_docx_keeps_document_order_tables_and_header():
    data = make_docx(["WORK EXPERIENCE", "Python Developer"], [["Jan 2020 – Mar 2022", "Acme"]],
                     header="Jane Doe")
    text = extract_document(data, "docx").text
    assert text.index("Jane Doe") < text.index("WORK EXPERIENCE") < text.index("Jan 2020") < text.index("AFTER TABLE")


def test_docx_zip_bomb_rejected():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("word/document.xml", "<w:document/>")
        z.writestr("word/big.bin", b"\0" * (60 * 1024 * 1024))
    with pytest.raises(ExtractionError, match="zip bomb"):
        extract_document(buf.getvalue(), "docx")


def test_pdf_page_cap(monkeypatch):
    monkeypatch.setattr(extractor, "MAX_PDF_PAGES", 3)
    monkeypatch.setattr(extractor, "OCR_ENABLED", False)
    res = extract_document(make_image_pdf(pages=6), "pdf")
    assert res.pages == 6
    assert any("first 3 of 6" in w for w in res.warnings)


@pytest.mark.parametrize("path", sample_cvs())
def test_all_sample_cvs_extract(path):
    res = extract_document(read(path), "pdf")
    assert len(res.text) > 1000
    assert res.ocr_pages == 0


def test_two_column_layout_keeps_columns_apart():
    found=[p for p in sample_cvs("HR Resume") if "Account-Manager" in p]
    if not found: pytest.skip("Original private resume not bundled")
    path = found[0]
    text = extract_document(read(path), "pdf").text
    assert "Regional Account Manager Dec 2018 - Aug 2022" in text


@pytest.mark.slow
def test_scanned_pdf_is_ocrd():
    if not extractor.ocr_available():
        pytest.skip("OCR not installed")
    found=[p for p in sample_cvs("AI CV's") if "Zohaib" in p]
    if not found: pytest.skip("Original private resume not bundled")
    src = found[0]
    scanned = make_image_pdf(text_pdf_page_source=read(src), scale=4.0)
    res = extract_document(scanned, "pdf")
    assert res.ocr_pages == 1
    assert "Zohaib" in res.text and "Software Engineering" in res.text
    assert any("OCR" in w for w in res.warnings)


@pytest.mark.slow
def test_scanned_cv_scores_like_the_text_version(jd_texts):
    if not extractor.ocr_available():
        pytest.skip("OCR not installed")
    from modules.pipeline import evaluate_batch
    found=[p for p in sample_cvs("AI CV's") if "Zohaib" in p]
    if not found: pytest.skip("Original private resume not bundled")
    src = read(found[0])
    text_c = evaluate_batch(jd_texts["ai"], [("t.pdf", src)]).candidates[0]
    scan_c = evaluate_batch(jd_texts["ai"], [("s.pdf", make_image_pdf(text_pdf_page_source=src, scale=4.0))]
                            ).candidates[0]
    assert abs(text_c.score - scan_c.score) <= 5
    assert scan_c.degrees_display == text_c.degrees_display
    assert "🔍 OCR used" in scan_c.flags


def test_docx_text_boxes_and_content_controls():
    import io as _io

    from docx import Document
    from lxml import etree
    d = Document()
    d.add_paragraph("MAIN BODY START")
    W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"
    sdt = etree.fromstring(
        f'<w:sdt xmlns:w="{W}"><w:sdtContent><w:p><w:r><w:t>Skills: Python, Django</w:t></w:r></w:p>'
        f'</w:sdtContent></w:sdt>')
    textbox = etree.fromstring(
        f'<w:p xmlns:w="{W}" xmlns:mc="{MC}"><w:r><mc:AlternateContent>'
        f'<mc:Choice Requires="wps"><w:drawing><w:txbxContent><w:p><w:r><w:t>EDUCATION BS Computer Science 2022'
        f'</w:t></w:r></w:p></w:txbxContent></w:drawing></mc:Choice>'
        f'<mc:Fallback><w:pict><w:txbxContent><w:p><w:r><w:t>EDUCATION BS Computer Science 2022</w:t></w:r>'
        f'</w:p></w:txbxContent></w:pict></mc:Fallback></mc:AlternateContent></w:r></w:p>')
    body = d.element.body
    body.insert(len(body) - 1, sdt)
    body.insert(len(body) - 1, textbox)
    buf = _io.BytesIO()
    d.save(buf)
    text = extract_document(buf.getvalue(), "docx").text
    assert "Skills: Python, Django" in text
    assert text.count("EDUCATION BS Computer Science 2022") == 1
    assert text.index("MAIN BODY START") < text.index("Skills")
