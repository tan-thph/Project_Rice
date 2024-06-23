from picamera2 import Picamera2
import cv2

def main():
    # Initialize Picamera2
    picam2 = Picamera2()

    # Configure Picamera2 preview
    picam2.configure_preview(size=(240, 280))

    # Start preview
    picam2.start_preview()

    while True:
        # Capture frame from Picamera2
        frame = picam2.capture()

        # Convert frame from RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Display frame using OpenCV
        cv2.imshow('Frame', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    picam2.stop_preview()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
