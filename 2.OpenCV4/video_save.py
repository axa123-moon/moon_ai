import datetime
import cv2

capture = cv2.VideoCapture("Image/maraisland.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')    # 코덱이름 설정
record = False

while True:
    if(capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)):
        capture.open("Image/maraisland.mp4")

    ret, frame = capture.read()
    cv2.imshow("VideoFrame", frame)

    now = datetime.datetime.now().strftime("%d_%H-%M-%S")
    key = cv2.waitKey(33)

    if key == 27:   # 27 = ESC
        break
    elif key == 26: # 26 = Ctrl + Z
        print("캡쳐")
        cv2.imwrite("Image/Capture/" + str(now) + ".png", frame)    # Capture 폴더를 미리 만들어 주어야 함
    elif key == 24:  # 24 = Ctrl + X
        print("녹화 시작")
        record = True
        video = cv2.VideoWriter("Image/Capture/" + str(now) + ".avi", fourcc, 20.0, (frame.shape[1], frame.shape[0]))
    elif key == 3:   # 3 = Ctrl + C
        print("녹화 중지")
        record = False
        video.release()
        
    if record == True:
        print("녹화 중..")
        video.write(frame)

capture.release()
cv2.destroyAllWindows()
