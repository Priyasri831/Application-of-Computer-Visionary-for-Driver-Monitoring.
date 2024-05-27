import cv2
import numpy as np
import pandas as pd
from datetime import datetime

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Path to your video file
video_path = 'VBOX0005.avi'

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Failed to open the video file.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video

c = 0
blink_records = []

blink_start_time = None

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        print("Error: Failed to read frame from the video.")
        break
    
    # Increase speed
    for _ in range(30):
        ret, img = cap.read()
        if not ret:
            print("Error: Failed to read frame from the video.")
            break

    ret, img = cap.retrieve()  # Retrieve the frame
    if not ret:
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    eyes_closed = True
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) == 0:  # Eyes closed
            if blink_start_time is None:
                blink_start_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert to seconds
        else:
            eyes_closed = False
            if blink_start_time is not None:
                blink_end_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert to seconds
                blink_duration = blink_end_time - blink_start_time
                frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1  # Get the frame number
                blink_time = frame_number / fps  # Calculate time in seconds
                video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert to seconds
                blink_records.append({'Blink': c, 'Frame Number': frame_number, 'Blink Time (s)': blink_time, 'Video Time (s)': video_time, 'Blink Duration (s)': blink_duration})
                c += 1
                blink_start_time = None

    if eyes_closed:
        if blink_start_time is None:
            blink_start_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert to seconds
    else:
        if blink_start_time is not None:
            blink_end_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert to seconds
            blink_duration = blink_end_time - blink_start_time
            frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1  # Get the frame number
            blink_time = frame_number / fps  # Calculate time in seconds
            video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Convert to seconds
            blink_records.append({'Blink': c, 'Frame Number': frame_number, 'Blink Time (s)': blink_time, 'Video Time (s)': video_time, 'Blink Duration (s)': blink_duration})
            c += 1
            blink_start_time = None

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Convert the blink records list to a pandas DataFrame
df = pd.DataFrame(blink_records)

# Save DataFrame to an Excel file
output_excel_file = '13blinkoutput.xlsx'
df.to_excel(output_excel_file, index=False)
print("Blink records saved to:", output_excel_file)
