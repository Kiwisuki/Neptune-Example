import pytest

from src import data_manipulation


@pytest.fixture()
def get_string_list():
    return ['python', 'is', 'fun']


def test_reverse_strings(get_string_list):
    assert data_manipulation.reverse_strings(get_string_list) == ['nohtyp', 'si', 'nuf']


def test_find_even_numbers():
    assert data_manipulation.find_even_numbers([11, 12, 13, 14, 15]) == [12, 14]
