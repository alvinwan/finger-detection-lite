"""View finger tracker results live."""


from train import get_index_finger_predictor
import cv2


def main():
    cap = cv2.VideoCapture(0)

    # get emotion predictor
    predictor = get_index_finger_predictor()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # flip across vertical axis
        frame_h, frame_w, _ = frame.shape

        x, y = predictor(frame)
        cv2.circle(frame, (x, y), 3, color=(0, 0, 255), thickness=-1)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()