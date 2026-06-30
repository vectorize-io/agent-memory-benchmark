from boltons.strutils import under2camel


def test_acronyms():
    assert under2camel("http_response") == "HTTPResponse"
    assert under2camel("api_key") == "APIKey"
    assert under2camel("parse_url") == "ParseURL"
    assert under2camel("user_id") == "UserID"


def test_non_acronyms_unchanged():
    assert under2camel("user_name") == "UserName"
    assert under2camel("complex_tokenizer") == "ComplexTokenizer"
