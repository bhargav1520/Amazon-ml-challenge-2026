"""Unit tests for preprocessor.py"""

import sys
from pathlib import Path
import pytest
import pandas as pd

# Add business_entity_resolution root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessor import (
    strip_accents,
    clean_text,
    clean_business_name,
    clean_address,
    extract_numbers,
    preprocess_dataframe,
)


def test_strip_accents():
    assert strip_accents("Société Générale") == "Societe Generale"
    assert strip_accents("Café de Paris") == "Cafe de Paris"
    assert strip_accents("Bangalore") == "Bangalore"


def test_clean_business_name():
    # US / India variations
    assert "pvt ltd" in clean_business_name("Infosys Pvt. Ltd.")
    assert "inc" in clean_business_name("Apple Corporation")
    assert "and" in clean_business_name("Johnson & Johnson")

    # French variations
    assert "sarl" in clean_business_name("Dupont S.A.R.L.")
    assert "sas" in clean_business_name("L'Oreal S.A.S.")


def test_clean_address():
    assert "street" in clean_address("123 Main St.")
    assert "road" in clean_address("MG Rd, Suite 400")
    assert "boulevard" in clean_address("10 Blvd Saint-Germain")
    assert "rue" in clean_address("15 Rue de Rivoli")


def test_extract_numbers():
    nums = extract_numbers("Flat 402, Building 12, MG Road 560001")
    assert "402" in nums
    assert "12" in nums
    assert "560001" in nums


def test_preprocess_dataframe():
    df = pd.DataFrame({
        "entity_id": ["S1-1"],
        "business_name": ["Google LLC"],
        "business_address": ["1600 Amphitheatre Pkwy, Mountain View, CA 94043"],
        "country": ["US"],
    })
    df_clean = preprocess_dataframe(df)
    assert "clean_name" in df_clean.columns
    assert "clean_address" in df_clean.columns
    assert "combined_text" in df_clean.columns
    assert "94043" in df_clean.iloc[0]["address_numbers"]
