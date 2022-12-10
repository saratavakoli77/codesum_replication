from affinity_data.tests.test_sacreablue import toy_data


def test_filter():
    r = toy_data.filter(lambda _: True)
    assert list(toy_data.iter_comments()) == list(r.iter_comments())
    assert r.projects[0] == r.projects[0].classes[0].project


def test_filter2():
    r = toy_data.filter(lambda method: method.method_name == "foo")
    assert list(toy_data.iter_comments()) != list(r.iter_comments())
    assert list(r.iter_comments()) == ["hello there"]
