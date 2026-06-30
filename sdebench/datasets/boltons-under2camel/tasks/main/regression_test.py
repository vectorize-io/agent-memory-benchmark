from boltons.strutils import under2camel


def test_http_acronym():
    assert under2camel("http_response") == "HTTPResponse"
