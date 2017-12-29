"""Collect data for finger tracker. Red dot follows regular path down.

Moves a red dot around the screen. Palm facing the screen, wrap your hand into
a fist, then extend your index finger. Using your index finger, point up. Then,
trace the red dot on the screen with your index finger.

We recommend rolling your sleeve down, so your wrist and arm are exposed to the
camera.
"""


import cv2
import random
import os
import time


RED = (0, 0, 255)


def main():

    # setup folder
    t0 = last = pause_until = time.time()
    folder = str(t0)[-4:]
    os.makedirs('data/%s' % folder, exist_ok=True)

    # Capture the first frame.
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    i = 0

    # movement specification
    speed = 3
    dx = dy = 1
    padding = 100
    padding_warning = 150
    min_duration_between_frames = 0.5  # in seconds

    # movement initialization
    h, w, _ = frame.shape
    x = padding + 10
    y = padding + 10

    pause_until = t0 + 5

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # flip across vertical axis
        t1 = time.time()

        # save image with label, once per second
        if t1 - last > min_duration_between_frames and t1 > pause_until:
            frame_to_draw = cv2.resize(frame, (w // 8, h // 8))
            last = time.time()
            side = 'right' if x > w // 2 else 'left'
            info = (folder, side, i, x // 8, y // 8)
            i += 1
            filename = 'data/%s/%s_%06d_x_%d_y_%d.png' % info
            cv2.imwrite(filename, frame_to_draw)

        # pause at the center of the image to switch hands
        if abs(x - w / 2) < 2:
            x += dx * 6
            pause_until = t1 + 3

        # move and draw dot
        if t1 > pause_until:
            x += dx * speed
        cv2.circle(frame, (x, y), 5, (0, 0, 255), thickness=-1)

        # Warning before changing direction
        if x <= padding_warning or x >= w - padding_warning or \
            y <= padding_warning or y >= h - padding_warning:
            cv2.imshow('frame', frame)  # hack
            cv2.addText(frame, "changing direction", (x+50, y), "Calibri",
                        color=RED)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # bounce dot away from the edge
        if x <= padding or x >= w - padding:
            dx *= -1
            if y - padding <= padding:
                dy = 1
            if y + padding >= h - padding:
                dy = -1
            y += dy * padding

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()