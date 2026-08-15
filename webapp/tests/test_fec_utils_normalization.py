from webapp.parser.utils import fec_utils


def test_party_normalize():
    assert fec_utils.party_normalize('DEM') == 'DEM'
    assert fec_utils.party_normalize('GOP') == 'REP'
    assert fec_utils.party_normalize('ind') == 'IND' or fec_utils.party_normalize('ind') == 'OTHER'


def test_money_and_date_normalize():
    assert fec_utils.money_normalize('1,234.56') == 1234.56
    assert fec_utils.money_normalize('(1,000)') == -1000.0
    assert fec_utils.date_normalize('04/03/2024') == '2024-04-03'
