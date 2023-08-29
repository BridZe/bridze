import re
import Levenshtein as Lev


def normalize_string(string):
    string = string.replace(' ', '')
    string = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\]^_\`{|}~\\\\]', '', string)
    return string


def compute_crr(answer, target):
    answer = normalize_string(answer)
    target = normalize_string(target)
    dist = Lev.distance(answer, target)
    length = len(answer)
    return f"{round((1-(dist/length))*100,2)}"
