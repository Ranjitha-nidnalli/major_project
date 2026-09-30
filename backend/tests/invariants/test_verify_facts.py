"""
knowledge/verify_facts.py: the mechanical checks that keep draft fact
records tied to their source text. Uses small inline fixtures, not the
gitignored converted corpus.
"""
from knowledge.verify_facts import normalize, numbers, split_pages, check_record

CONVERTED = (
    "######## demo | pdf_page_index=5 | printed_page=10\n"
    "ಕ್ಲೋರ್‍ಪೈರಿಫಾಸ್ 20 ಇ.ಸಿ\n2 ಮಿ.ಲೀ\n600   ಮಿ.ಲೀ\n"
    "######## demo | pdf_page_index=6 | printed_page=11\n"
    "ಎಕರೆಗೆ 10,000 ದಿಂದ 14,000 ಮೂರು ಕಣ್ಣಿನ ತುಂಡುಗಳು\n"
)
PAGES = split_pages(CONVERTED)


def _record(**kw):
    base = {"id": "r", "pdf_page_index": 5, "text_kn": "", "excerpts": ["2 ಮಿ.ಲೀ"]}
    base.update(kw)
    return base


def test_normalize_drops_zero_width_joiners_and_collapses_space():
    assert normalize("ಕ್ಲೋರ್‍ಪೈ  ರಿ\n") == "ಕ್ಲೋರ್ಪೈ ರಿ"


def test_numbers_strip_thousands_separators():
    assert numbers("10,000 ದಿಂದ 14,000; 2.5 ಗ್ರಾಂ") == {"10000", "14000", "2.5"}


def test_split_pages_keys_by_pdf_page_index():
    assert set(PAGES) == {5, 6}
    assert "600 ಮಿ.ಲೀ" in PAGES[5]


def test_record_passes_when_excerpt_and_numbers_match():
    r = _record(text_kn="ಲೀಟರ್‌ಗೆ 2 ಮಿ.ಲೀ, ಎಕರೆಗೆ 600 ಮಿ.ಲೀ", excerpts=["2 ಮಿ.ಲೀ 600 ಮಿ.ಲೀ"])
    assert check_record(r, PAGES) == []


def test_excerpt_matches_across_a_zero_width_joiner():
    assert check_record(_record(excerpts=["ಕ್ಲೋರ್ಪೈರಿಫಾಸ್ 20 ಇ.ಸಿ"]), PAGES) == []


def test_number_not_in_source_is_caught():
    """The failure this guards against: a dose typed wrongly (6000 for 600)."""
    r = _record(per_acre="6000 ಮಿ.ಲೀ", excerpts=["2 ಮಿ.ಲೀ 600 ಮಿ.ಲೀ"])
    assert any("6000" in p for p in check_record(r, PAGES))


def test_excerpt_from_the_wrong_page_is_caught():
    r = _record(pdf_page_index=6, excerpts=["2 ಮಿ.ಲೀ"])
    assert any("not found" in p for p in check_record(r, PAGES))


def test_multi_page_record_accepts_excerpts_from_either_page():
    r = _record(pdf_page_indexes=[5, 6], excerpts=["2 ಮಿ.ಲೀ", "10,000 ದಿಂದ 14,000"],
                text_kn="10,000 ದಿಂದ 14,000")
    assert check_record(r, PAGES) == []


def test_record_without_excerpts_fails():
    assert "no excerpts" in check_record(_record(excerpts=[]), PAGES)
