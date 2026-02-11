from src.preprocessing import clean_text

def test_clean_text_lowercase():
    # Test if it actually lowercases text
    assert clean_text("HELLO World") == "hello world"

def test_clean_text_punctuation():
    # Test if it removes punctuation
    assert clean_text("Fake News!!!") == "fake news"