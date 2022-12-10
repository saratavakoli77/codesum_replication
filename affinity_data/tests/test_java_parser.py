import javalang
from unittest.mock import MagicMock

from affinity_data.data_representations import ScrappedProject, ScrappedClass, ScrappedMethod
from affinity_data.java_parser import parse_text

parse_string = """
    package javalang.brewtab.com; 
    class Test { 
        /** Here are comments **/ 
        public static void print() {
            system.out.println(\"Hello world\"); 
            system.out.println(\"foo\");
        } 
        /** Here are new comments! **/ 
        public static void print(String s) { 
            System.out.println(\"Hello world new\"); 
        } 
    }
"""


def test_javalang():
    javalang.parse.parse(parse_string)


def test_parse_text():
    project = MagicMock()
    cls = parse_text(parse_string, "Foo", "Foo.java", project)
    assert isinstance(cls, ScrappedClass)
    assert len(cls.methods) == 2
    m1, m2 = cls.methods
    assert isinstance(m1, ScrappedMethod)
    assert m1.class_name == "Foo"
    assert m1.comment == "/** Here are comments **/"
    assert m2.comment == "/** Here are new comments! **/"

