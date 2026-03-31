import pytest

from webapp.parser.Context_Integration.Context_Library import constants as C


def test_get_party_code_info_known():
    info = C.get_party_code_info("DEM")
    assert isinstance(info, dict)
    assert "democratic" in info.get("description", "").lower()


def test_normalize_party_code_variants():
    dem_code = C.normalize_party_code("DEM")
    assert dem_code is not None and "democratic" in dem_code.lower()
    
    dem_lower = C.normalize_party_code("dem")
    assert dem_lower is not None and "democratic" in dem_lower.lower()
    
    d_code = C.normalize_party_code("D")
    assert d_code is not None and "democratic" in d_code.lower()
    
    w_code = C.normalize_party_code("W")
    assert w_code is not None and "write" in w_code.lower()
    
    dc = C.normalize_party_code("D/C")
    assert dc is not None and "democratic" in dc.lower() and "conserv" in dc.lower()


def test_get_party_code_info_unknown():
    assert C.get_party_code_info("ZZZ") is None
