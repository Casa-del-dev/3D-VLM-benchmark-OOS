import cv2

video_path = "/Users/fangzhouma/Desktop/3d_vision/3D-VLM-benchmark-OOS/data/HD-EPIC/Videos/P01/P01-20240203-184045.mp4"
frame_number = 6*30  # change this to the frame you want

cap = cv2.VideoCapture(video_path)

# Set the frame position
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

ret, frame = cap.read()

if ret:
    cv2.imwrite("data/for_debug/time_6.jpg", frame)
    print("Frame saved!")
else:
    print("Failed to extract frame.")

cap.release()