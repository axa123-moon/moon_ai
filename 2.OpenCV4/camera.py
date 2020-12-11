import cv2

# 프로그램 실행전에 카메라를 활성화 시켜주어야 함
capture = cv2.VideoCapture(0)   # 기본적으로 장치번호는 0, 여러대가 있을 경우 선택할 수 있음
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = capture.read()
    cv2.imshow("VideoFrame", frame)
    #if cv2.waitKey(1) > 0: break
    if cv2.waitKey(33) == ord('q'): break   # 'q'문자가 입력되면 종료
    
capture.release()
cv2.destroyAllWindows()