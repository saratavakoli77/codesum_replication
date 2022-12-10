"""
File to parse Java ASTs and return method comments.
"""

import javalang
from javalang.parser import JavaParserBaseException
from pathlib import Path
import sys
import chardet
from typing import Union

from affinity_data.data_representations import UnexpectedUnparseableSourceError, \
    Java9ModuleError, JavaSyntaxError, ScrappedProject, FailedScrappedClass, JavaBaseError, \
    IgnoredClass, ScrappedClass, ScrappedMethod, ClassLike, JavaRecursionLimit, FileReadError

MethodDeclaration = javalang.tree.MethodDeclaration


def parse_text(
    file_text: str,
    class_name: str,
    file_path: str,
    project: ScrappedProject,
    print_failing_files: bool = False
) -> ClassLike:
    # See anything obviously wrong with the file text
    assert isinstance(file_text, str)
    if not file_text_passes_heuristic_filters(file_text):
        return IgnoredClass(project, file_path, class_name)

    # Define some methods for handling if things go wrong
    def print_bad_text():
        if print_failing_files:
            print("Bad text", file_path, class_name, project.project_name)
            print(file_text)

    def build_fail(exception: UnexpectedUnparseableSourceError):
        return FailedScrappedClass(
            project, file_path, class_name, exception
        )

    # Attempt to actually parse the thing
    try:
        tree = javalang.parse.parse(file_text)
    except AttributeError as e:
        print_bad_text()
        return build_fail(UnexpectedUnparseableSourceError())
    except javalang.parser.JavaSyntaxError as e:
        if file_text_looks_like_includes_java_module(file_text):
            build_fail(Java9ModuleError())
            #return None  # Failing on java 9 modules is expected
        print_bad_text()
        return build_fail(JavaSyntaxError())
    except javalang.parser.JavaParserBaseException as e:
        print_bad_text()
        return build_fail(JavaBaseError())
    except Exception as e:
        return build_fail(UnexpectedUnparseableSourceError())

    # Return a object from this parsed tree
    cls = ScrappedClass(project, methods=[], file_name=file_path, class_name=class_name)
    try:
        add_methods_from_java_tree(cls, tree)
    except RecursionError as e:
        return build_fail(JavaRecursionLimit())
    return cls


def parse_file_and_add_to_project(
    class_name: str,
    file_path: str,
    project: ScrappedProject,
    print_failing_files: bool = False
) -> None:
    try:
        file_text = read_text_unkown_encoding(Path(file_path))
    except (UnicodeDecodeError, LookupError):
        project.add_class_like(FailedScrappedClass(project, file_path, class_name, FileReadError()))
        return None
    new_class = parse_text(file_text, class_name, file_path, project, print_failing_files)
    project.add_class_like(new_class)


def read_text_unkown_encoding(file_path: Path) -> str:
    file_bytes = file_path.read_bytes()
    encoding_detect = chardet.detect(file_bytes)
    text_encoding = encoding_detect['encoding']
    if text_encoding is None:
        # Not sure why this happens. Just go with utf-8 as a default
        text_encoding = 'utf-8'
    file_text = file_bytes.decode(text_encoding)
    return file_text


def add_methods_from_java_tree(add_to_class: ScrappedClass, ast_tree) -> None:
    """Mutates a ScrappedClass to add a new method taken from a parsed ast. The
    parsed ast is supposed to come from the javalang module."""
    sys.setrecursionlimit(3000)
    for path, node in ast_tree:
        if (isinstance(node, MethodDeclaration)):  # check for method
            method_name = node.name
            assert isinstance(method_name, str)
            method_comment = node.documentation
            if method_comment is not None:
                method_comment = method_comment.strip()
            add_to_class.methods.append(
                ScrappedMethod(
                    method_name=method_name,
                    method_class=add_to_class,
                    comment=method_comment
                )
            )


# Helper methods during parse

def file_text_passes_heuristic_filters(file_text: str):
    return file_text_looks_auto_generated(file_text)


def file_text_looks_auto_generated(file_text: str):
    """A weak heuristic to see if a file text is auto generated. We do this
    just by looking for the word 'generated' in the first few lines of the
    file."""
    file_lines = file_text.split("\n")
    num_lines_to_look_at = min(3, len(file_lines))
    for line in file_lines[:num_lines_to_look_at]:
        if "generated" in line.lower():
            return False
    return True


def file_text_looks_like_includes_java_module(file_text: str) -> bool:
    """java modules are a Java 9 feature where as this javalang module we are
    using is only up to version 8. This uses a weak heuristic to see if a file
    includes a java module"""
    file_lines = file_text.split("\n")

    def line_looks_like_a_module_declaration(line: str):
        words = line.split()
        return words and words[0] == "module" and line.strip().endswith("{")
    return any(
        line_looks_like_a_module_declaration(line)
        for line in file_lines
    )

