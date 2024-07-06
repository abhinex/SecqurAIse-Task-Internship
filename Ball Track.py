import cv2
import numpy as np
import time

# Initialize variables
time_events = {color: [] for color in ["yellow", "orange", "green", "white"]}
quad_events = {color: [] for color in ["yellow", "orange", "green", "white"]}
start_time = time.time()
output_file = "events_log.txt"

# Color ranges for detecting different colored balls
color_ranges = {
    "yellow": ((20, 100, 100), (30, 255, 255)),
    "orange": ((5, 100, 100), (15, 255, 255)),
    "green": ((35, 100, 100), (85, 255, 255)),
    "white": ((0, 0, 200), (180, 20, 255)),
}

def detect_quadrant(x, y, frame_width, frame_height):
    w_half, h_half = frame_width // 2, frame_height // 2
    if x > w_half and y > h_half: return 1
    elif x <= w_half and y > h_half: return 2
    elif x <= w_half and y <= h_half: return 3
    elif x > w_half and y <= h_half: return 4

def log_event(color, timestamp, quadrant, event_type):
    with open(output_file, "a") as f:
        f.write(f"{timestamp:.2f}, {quadrant}, {color}, {event_type}\n")

def process_frame(frame, frame_width, frame_height):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 2000:
                x, y, w, h = cv2.boundingRect(contour)
                center_x, center_y = x + w // 2, y + h // 2
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, color, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

                current_quad = detect_quadrant(center_x, center_y, frame_width, frame_height)
                timestamp = time.time() - start_time
                if not time_events[color] or timestamp - time_events[color][-1] > 3:
                    time_events[color].append(timestamp)
                    quad_events[color].append(current_quad)
                    log_event(color, timestamp, current_quad, "Entry")
                else:
                    time_events[color][-1] = timestamp
                    quad_events[color][-1] = current_quad
                    log_event(color, timestamp, current_quad, "Exit")

    return frame

def main():
    cap = cv2.VideoCapture('D:/AI Assignment video.mp4')
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video", frame_width, frame_height)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, frame_width, frame_height)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()