# 이 코드는 주어진 파일을 Hugging Face 모델 API에 전송하여 처리한 결과를 받아오는 기능을 가진 함수를 보여줍니다. 
# 입력 파일의 데이터를 POST 요청을 통해 API에 전송하고, API에서 반환된 JSON 형태의 응답 데이터를 반환합니다.


import requests

# Hugging Face 모델 API 설정
API_URL = "https://api-inference.huggingface.co/models/oceanstar/bridze"
headers = {"Authorization": "Bearer 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"}


# Hugging Face 모델 API에 파일을 보내고 처리된 결과를 받아오는 역할을 하는 함수
def query(filename):
    with open(filename, "rb") as f:
        data = f.read()  # 파일 데이터 읽기
    response = requests.post(API_URL, headers=headers, data=data)  # POST 요청으로 파일 데이터 전송
    return response.json()  # JSON 형태로 응답 데이터 반환

