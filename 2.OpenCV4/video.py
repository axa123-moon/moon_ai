import cv2

capture = cv2.VideoCapture("Image/maraisland.mp4")

while True:
    if(capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)):
        # CAP_PROP_POS_FRAMES : 현재 프레임의 수, CAP_PROP_FRAME_COUNT : 총 프레임의 수
        capture.open("Image/maraisland.mp4")

    ret, frame = capture.read()     # 프레임을 하나씩 읽음
    cv2.imshow("VideoFrame", frame)

    if cv2.waitKey(33) > 0: break   # 33ms 간격으로 기다림

capture.release()
cv2.destroyAllWindows()
