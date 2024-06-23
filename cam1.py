import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("open camera failed")
    exit()

while True:
    
    ret, frame = cap.read()

    if ret:
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("read failed")
        break

cap.release()
cv2.destroyAllWindows()
