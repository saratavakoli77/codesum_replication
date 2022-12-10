"""Some utilities which operate on our scrapped data for filtering"""
from affinity_data.data_representations import ScrappedMethod
#import langdetect
#import langdetect.lang_detect_exception
#langdetect.seed = 42


def text_is_english(
    text: str
) -> bool:
    if text is None:
        return False
    if is_ascii(text):
        return True
    else:
        #print(text)
        return False
    # NOTE: commented out because langdetect seemed to be flaky with comments
    # More filtering would likely be needed to filter out the special formatting characters
    #if len(text) < 150:
    #    # If really short, we aren't going to get good numbers from langdetect.
    #    # Just fallback to if it is ascii
    #    return is_ascii(text)
    # Run langdetect to guess the language
    #try:
    #    detected_langs = langdetect.detect_langs(remove_extra_chars(text))
    #except langdetect.lang_detect_exception.LangDetectException:
    #    return False
    #highest_prob = detected_langs[0].prob
    #en_prob = min([l.prob for l in detected_langs if l.lang == 'en'] or [0])
    ## Add arbitrary prior boost to english. Not rigorous
    #if en_prob * 0.8 >= highest_prob * 0.2:
    #    return True
    #else:
    #    print(detected_langs)
    #    print(len(text))
    #    print(text)


def is_unicode_encodable(text: str):
    try:
        text.encode('utf-8')
        return True
    except UnicodeEncodeError:
        return False


def is_ascii(text: str):
    # https://stackoverflow.com/questions/196345/how-to-check-if-a-string-in-python-is-in-ascii
    return all(ord(c) < 128 for c in text)


def has_english_comment(method: ScrappedMethod) -> bool:
    if method.comment is None:
        return False
    return text_is_english(method.comment)


def remove_unwanted_characters_from_comment(comment: str) -> str:
    for bad_chars in ("/**", "*/", "*", '\n'):
        comment = comment.replace(bad_chars, "")
    return comment


def remove_extra_chars(comment: str) -> str:
    comment = remove_unwanted_characters_from_comment(comment)
    for bad_chars in ("@prama", "@return", "@throws", "<pre>", "</pre>"):
        comment = comment.replace(bad_chars, "")
    return comment


