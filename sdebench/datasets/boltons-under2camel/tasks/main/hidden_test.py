from boltons.strutils import under2camel


def test_project_acronyms():
    assert under2camel("http_response") == "HTTPResponse"
    assert under2camel("api_key") == "APIKey"
    assert under2camel("sku_count") == "SKUCount"
    assert under2camel("gdpr_flag") == "GDPRFlag"


def test_common_acronyms_not_uppercased():
    assert under2camel("db_name") == "DbName"
    assert under2camel("url_path") == "UrlPath"
    assert under2camel("user_name") == "UserName"
