from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class Document:
    comment: str
    code_words: str
    doc_id: str
    split_name: str


def get_all_example_pairs(
    code_word_file: Path,
    comment_file: Path,
    split_name: str,
    has_id: bool = True,
    id_is_tsv: bool = False,
    should_remove_special_tags: bool = True,
    dataset_name: str = None  # DON"T USE UNTIL UPDATEING CALLERS
) -> Iterable[Document]:
    code_text, comment_text = \
        code_word_file.read_text().split("\n"), comment_file.read_text().split("\n")
    # assert len(code_text) == len(comment_text)

    def get_text_with_id(text):
        comma_idx = text.index("\t" if id_is_tsv else ",")
        id, text = text[:comma_idx], text[comma_idx + 1:]
        return id, text.strip()

    def actually_gen():
        for i, (code, comment) in enumerate(zip(code_text, comment_text)):
            if not code.strip() or not comment.strip():
                continue  # weirdly sometime get a thing with no text? Ignore this.

            def parse_line(line):
                if has_id:
                    try:
                        id, val = get_text_with_id(line)
                    except:
                        raise ValueError("bad line", i, line)
                else:
                    val = line.strip()
                    id = f"{split_name}{i}"
                return id, val
            code_id, code = parse_line(code)
            comment_id, comment = parse_line(comment)
            assert code_id == comment_id
            #if dataset_name == "codenn":
            #    # Super quick hack fix
            #    terminal = "<EOF>"
            #    assert code.endswith(terminal), code
            #    code = code[: -1*(len(terminal) + 1)]
            if should_remove_special_tags:
                comment = " ".join(remove_special_tags(comment.split(), dataset_name))
                code = " ".join(remove_special_tags(code.split(), dataset_name))
            if code == "":
                continue  # Sometimes only a special token
            yield Document(
                comment=comment,
                code_words=code,
                doc_id=str(code_id),
                split_name=split_name
            )

    # Wrap in a class so the iterable has a __len__
    class DataProcAllExamplesWrapper:
        def __len__(self):
            return len(code_text)

        def __iter__(self):
            return actually_gen()

    return DataProcAllExamplesWrapper()


def remove_special_tags(tokens: List[str], dataset_name: str = None):
    """remove tokens like <s> and </s> but not <code>foo</code>"""
    def is_special(t):
        return t[0] == "<" and t[-1] == ">" and ">" not in t[1:-1]

    return [
        t for t in tokens
        if not is_special(t)
    ]
