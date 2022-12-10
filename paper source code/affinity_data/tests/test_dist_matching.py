# We are not currently using the dist matcher

#from affinity_data.dist_matching import *
#
#
#def test_matcher_meth_name():
#    matcher = MethodMatcher([
#        Method("Foobar"),
#        Method("barbaz"),
#        Method("POPbarbaz"),
#        Method("Asdfwosdf"),
#        Method("abcdefg"),
#    ])
#    r = matcher.method_names_by_edit_dist(Method("foobar"))
#    top_m, top_d = r[0]
#    assert top_m.method_name == "Foobar"
#    assert top_d == 0
#
#    r = matcher.method_names_by_edit_dist(Method("foobab"))
#    top_m, top_d = r[0]
#    assert top_m.method_name == "Foobar"
#    assert top_d == 1
#
#    r = matcher.method_names_by_edit_dist(Method("araz"))
#    top_m, top_d = r[0]
#    assert top_m.method_name == "barbaz"
#    assert top_d == 2
#
#
#def test_what():
#    assert editdistance.eval("invoke", "getAndAccumulate") != 3
