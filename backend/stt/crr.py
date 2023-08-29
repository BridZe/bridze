# 이 코드는 주어진 두 문자열 간의 유사도를 Levenshtein 거리를 활용하여 계산합니다.
# 입력 문자열은 공백 및 특수문자를 제거하여 정규화한 후 유사도를 계산하며, 유사도는 백분율로 나타냅니다.


import re
import Levenshtein as Lev


# 문자열 정규화 함수
def normalize_string(string):
    string = string.replace(' ', '')  # 공백 제거
    string = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\]^_\`{|}~\\\\]', '', string)  # 특수문자 제거
    return string


# 문자열 유사도 계산 함수
def compute_crr(answer, target):
    answer = normalize_string(answer)  # 정규화된 형태로 변환
    target = normalize_string(target)
    dist = Lev.distance(answer, target)  # Levenshtein 거리 계산
    length = len(answer)  # 정규화된 문자열의 길이
    return f"{round((1-(dist/length))*100,2)}"  # 정확도 계산하여 반환
