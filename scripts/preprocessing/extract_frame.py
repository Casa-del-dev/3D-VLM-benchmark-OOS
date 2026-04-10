import cv2

video_path = "/Users/fangzhouma/Desktop/3d_vision/3D-VLM-benchmark-OOS/data/HD-EPIC/Videos/P04/P04-20240413-142619.mp4"
frame_number = 208*30  # change this to the frame you want

cap = cv2.VideoCapture(video_path)

# Set the frame position
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

ret, frame = cap.read()

if ret:
    cv2.imwrite("frame_208.jpg", frame)
    print("Frame saved!")
else:
    print("Failed to extract frame.")

cap.release()