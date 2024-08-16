import cv2
import numpy as np
from pygame import mixer
from twilio.rest import Client
import time
import threading

# Function to make a call using Twilio
def make_call(to, url):
    account_sid = 'Enter your accounts sid'  # Replace with your Twilio Account SID
    auth_token = 'Enter your accounts auth token'    # Replace with your Twilio Auth Token
    client = Client(account_sid, auth_token)

    call = client.calls.create(
        to=to,
        from_='Enter the phone number provided by twilio',  # Replace with your Twilio phone number
        url=url
    )
    print(f"Call initiated: {call.sid}")

# Function to detect fire
def detect_fire(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Refined HSV thresholds for fire detection
    lower_bound = np.array([0, 100, 200], dtype=np.uint8)
    upper_bound = np.array([25, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Apply morphological operations to filter out noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Brightness threshold to avoid false positives from overexposed regions
    brightness = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(brightness, 200, 255, cv2.THRESH_BINARY)
    combined_mask = cv2.bitwise_and(mask, cv2.bitwise_not(bright_mask))

    # Find contours in the combined mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter based on contour area to reduce false positives
    min_contour_area = 500
    max_contour_area = 10000  # Upper limit to avoid large overexposed regions
    for contour in contours:
        if min_contour_area < cv2.contourArea(contour) < max_contour_area:
            return True

    return False

# Function to handle video capture and processing
def process_video():
    cap = cv2.VideoCapture(0)  # Replace 0 with the appropriate camera index

    alarm_triggered = False
    last_detection_time = 0
    detection_threshold = 30  # Seconds before allowing another alarm
    consecutive_frames = 0
    required_consecutive_frames = 60  # Number of consecutive frames with fire detection

    mixer.init()  # Initialize the mixer
    alarm_sound_path = r'Enter the file path of the WAV file provided'

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))

        if detect_fire(frame):
            consecutive_frames += 1
            if consecutive_frames >= required_consecutive_frames:
                current_time = time.time()
                if not alarm_triggered or (current_time - last_detection_time > detection_threshold):
                    print("Fire detected! Triggering alarm and making a phone call.")
                    mixer.music.load(alarm_sound_path)
                    mixer.music.play()
                    make_call("Enter the phone number you registered in twilio", "Make a twiml bin message and enter the url here")
                    alarm_triggered = True
                    last_detection_time = current_time
                    consecutive_frames = 0  # Reset the frame counter after the call
        else:
            consecutive_frames = 0

        # Show the processed video feed
        cv2.imshow('Fire Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function to start video processing in a separate thread
if __name__ == "__main__":
    video_thread = threading.Thread(target=process_video)
    video_thread.start()
