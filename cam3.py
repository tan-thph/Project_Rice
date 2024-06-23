import cv2

def main():
    # Try to read an image
    img_path = '/home/pi/Downloads/1.jpeg'
    img = cv2.imread(img_path)
    if img is None:
        print(f'Failed to load image from {img_path}')
    else:
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Open camera feed
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Failed to open camera')
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Failed to grab frame')
            break

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
