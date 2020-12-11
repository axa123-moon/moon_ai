import numpy as np
import cv2

src = cv2.imread('Image/office.jpg')
src2 = cv2.cvtColor(src, cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(src2, 127, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

kernel2 = np.array([[1, 0, 0, 0, 1],
                    [0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 0],
                    [0, 1, 0, 1, 0],
                    [1, 0, 0, 0, 1]])

open = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel, iterations=9)
close = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel, iterations=9)
#hit = cv2.morphologyEx(src, cv2.MORPH_HITMISS, kernel, iterations=9)
# 생성된 구조 요소(kernel)를 활용해 모폴로지 변환을 적용합니다.
# 모폴로지 함수(cv2.morphologyEx)로 모폴로지 연산을 진행합니다.
# cv2.morphologyEx(원본 배열, 연산 방법, 구조 요소, 고정점, 반복 횟수, 테두리 외삽법, 테두리 색상)로 모폴로지 연산을 진행합니다.
# 연산 방법에 따라, 모폴로지 연산 결과가 달라집니다

# cv2.MORPH_DILATE 팽창 연산 
# cv2.MORPH_ERODE 침식 연산 
# cv2.MORPH_OPEN 열림 연산 
# cv2.MORPH_CLOSE 닫힘 연산 
# cv2.MORPH_GRADIENT 그레이디언트 연산 
# cv2.MORPH_TOPHAT 탑햇 연산 
# cv2.MORPH_BLACKHAT 블랙햇 연산 
# cv2.MORPH_HITMISS 히트미스 연산 

dst = np.concatenate((src, open, close), axis=1)

dst = cv2.resize(dst, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)    # 이미지 크기 축소

cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()