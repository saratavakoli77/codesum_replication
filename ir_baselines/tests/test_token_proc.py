from data_handling.token_proc import *


def test_camel_case():
    assert camel_case_split("fooBarBaz") == ["foo", "Bar", "Baz"]
    assert camel_case_split("FooBarBaz") == ["Foo", "Bar", "Baz"]
    assert camel_case_split("FOOBarBaz") == ["FOO", "Bar", "Baz"]


def test_non_alphanum():
    assert tokenize_non_alphanum("foobar") == ["foobar"]
    assert tokenize_non_alphanum("foo_bar;") == ["foo", "_", "bar", ";"]
    assert tokenize_non_alphanum("foo_bar;", ["_"]) == ["foo_bar", ";"]
