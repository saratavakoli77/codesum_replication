"""
From https://similarapi.appspot.com/allLibPair.html

Unused Code
"""
from dataclasses import dataclass


@dataclass
class PPair:
    """Project Pair"""
    proj1: str
    proj2: str
    proj1_subpath: str = None
    proj2_subpath: str = None


similar_api_pairs = [
    PPair("spring-projects/spring-security-oauth", "thymeleaf/thymeleaf"),  # thymeleaf?
    PPair("easymock/easymock", "mockito/mockito"),
    PPair("jmock-developers/jmock-library", "powermock/powermock"),
    PPair("sgothel/jogl", "LWJGL/lwjgl"),
    PPair("spring-projects/spring-ldap", "pingidentity/ldapsdk"),
    PPair("hvtuananh/lingpipe", "apache/opennlp"),
    PPair("gephi/gephi", "jrtom/jung"),
    PPair("apache/mina", "jpcap/jpcap"),
    PPair("halfhp/androidplot", "julienchastang/charts4j"),
    # PPair("apache/lucene-solr", ""), # ?? lucene, solr
    # PPair("", ""), ?? awt, swing
    # java-3d, jogl
    # lingpipe, stanford-nlp
    # hamcrest, jmockit
    PPair("Netflix/astyanax", "datastax/java-driver"),
]