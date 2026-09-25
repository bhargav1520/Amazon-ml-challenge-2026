"""Memory-optimized Text Normalization for Business Entity Resolution.
Processes string cleaning efficiently without unnecessary memory overhead.
"""

import re
import unicodedata
from typing import Dict, List, Set, Tuple
import pandas as pd


# Precompiled regex patterns for speed and low memory
LEGAL_SUFFIXES_RAW = [
    (re.compile(r"\bprivate\s+limited\b", re.I), "pvt ltd"),
    (re.compile(r"\bpvt\.?\s*ltd\.?\b", re.I), "pvt ltd"),
    (re.compile(r"\bpvt\s*limited\b", re.I), "pvt ltd"),
    (re.compile(r"\bcorporation\b", re.I), "inc"),
    (re.compile(r"\bcorp\.?\b", re.I), "inc"),
    (re.compile(r"\bincorporated\b", re.I), "inc"),
    (re.compile(r"\binc\.?\b", re.I), "inc"),
    (re.compile(r"\blimited\s+liability\s+company\b", re.I), "llc"),
    (re.compile(r"\bl\.?l\.?c\.?\b", re.I), "llc"),
    (re.compile(r"\blimited\s+liability\s+partnership\b", re.I), "llp"),
    (re.compile(r"\bl\.?l\.?p\.?\b", re.I), "llp"),
    (re.compile(r"\blimited\b", re.I), "ltd"),
    (re.compile(r"\bltd\.?\b", re.I), "ltd"),
    (re.compile(r"\bcompany\b", re.I), "co"),
    (re.compile(r"\bco\.?\b", re.I), "co"),
    (re.compile(r"\bs\.?a\.?r\.?l\.?\b", re.I), "sarl"),
    (re.compile(r"\bs\.?a\.?s\.?\b", re.I), "sas"),
    (re.compile(r"\bs\.?a\.?\b", re.I), "sa"),
    (re.compile(r"\be\.?u\.?r\.?l\.?\b", re.I), "eurl"),
    (re.compile(r"\bgmbh\b", re.I), "gmbh"),
]

ADDRESS_ABBREVIATIONS = [
    (re.compile(r"\bst\.?\b", re.I), "street"),
    (re.compile(r"\bstreet\b", re.I), "street"),
    (re.compile(r"\brd\.?\b", re.I), "road"),
    (re.compile(r"\broad\b", re.I), "road"),
    (re.compile(r"\bave\.?\b", re.I), "avenue"),
    (re.compile(r"\bavenue\b", re.I), "avenue"),
    (re.compile(r"\bav\.?\b", re.I), "avenue"),
    (re.compile(r"\bblvd\.?\b", re.I), "boulevard"),
    (re.compile(r"\bboulevard\b", re.I), "boulevard"),
    (re.compile(r"\bbd\.?\b", re.I), "boulevard"),
    (re.compile(r"\bdr\.?\b", re.I), "drive"),
    (re.compile(r"\bdrive\b", re.I), "drive"),
    (re.compile(r"\bln\.?\b", re.I), "lane"),
    (re.compile(r"\blane\b", re.I), "lane"),
    (re.compile(r"\bhwy\.?\b", re.I), "highway"),
    (re.compile(r"\bhighway\b", re.I), "highway"),
    (re.compile(r"\bfl\.?\b", re.I), "floor"),
    (re.compile(r"\bfloor\b", re.I), "floor"),
    (re.compile(r"\bste\.?\b", re.I), "suite"),
    (re.compile(r"\bsuite\b", re.I), "suite"),
    (re.compile(r"\bapt\.?\b", re.I), "apartment"),
    (re.compile(r"\bapartment\b", re.I), "apartment"),
    (re.compile(r"\bbldg\.?\b", re.I), "building"),
    (re.compile(r"\bbuilding\b", re.I), "building"),
    (re.compile(r"\bopp\.?\b", re.I), "opposite"),
    (re.compile(r"\bopposite\b", re.I), "opposite"),
    (re.compile(r"\bnr\.?\b", re.I), "near"),
    (re.compile(r"\bnear\b", re.I), "near"),
    (re.compile(r"\brue\b", re.I), "rue"),
    (re.compile(r"\ballee\b", re.I), "allee"),
    (re.compile(r"\bchemin\b", re.I), "chemin"),
    (re.compile(r"\bplace\b", re.I), "place"),
    (re.compile(r"\bpl\.?\b", re.I), "place"),
]

NON_ALPHANUM_RE = re.compile(r"[^\w\s]")
WHITESPACE_RE = re.compile(r"\s+")
DIGITS_RE = re.compile(r"\b\d+\b")


def strip_accents(text: str) -> str:
    """Removes diacritics and accents."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFD", text)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


def clean_text(text: str) -> str:
    """Lowercases, strips accents, normalizes symbols."""
    if not text or not isinstance(text, str):
        return ""
    text = strip_accents(text).lower()
    text = text.replace("&", " and ").replace("@", " at ").replace("/", " ").replace("-", " ")
    return text


def clean_business_name(name: str) -> str:
    """Normalizes business names by expanding legal entities and removing noise."""
    name = clean_text(name)
    for pattern, repl in LEGAL_SUFFIXES_RAW:
        name = pattern.sub(repl, name)
    name = NON_ALPHANUM_RE.sub(" ", name)
    return WHITESPACE_RE.sub(" ", name).strip()


def clean_address(address: str) -> str:
    """Normalizes addresses by expanding abbreviations."""
    address = clean_text(address)
    for pattern, repl in ADDRESS_ABBREVIATIONS:
        address = pattern.sub(repl, address)
    address = NON_ALPHANUM_RE.sub(" ", address)
    return WHITESPACE_RE.sub(" ", address).strip()


def extract_numbers(text: str) -> Set[str]:
    """Extracts all continuous digits (postal codes, building/suite numbers)."""
    if not text:
        return set()
    return set(DIGITS_RE.findall(text))


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """In-place, memory-efficient preprocessing of dataframe."""
    df["clean_name"] = [clean_business_name(x) for x in df["business_name"]]
    df["clean_address"] = [clean_address(x) for x in df["business_address"]]
    df["combined_text"] = df["clean_name"] + " " + df["clean_address"]
    df["address_numbers"] = [extract_numbers(x) for x in df["clean_address"]]
    return df
