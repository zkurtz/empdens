from empdens import data


def test_load_Japanese_vowels_data():
    df = data.load_Japanese_vowels_data()
    assert df.shape == (1456, 13)
