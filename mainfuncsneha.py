import cv2
import mediapipe as mp
import math

def drawing(img, landmarks):
    for landmark in landmarks:
        x, y, _ = landmark
        cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), -1)

def calculate_angle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    if angle < 0:
        angle += 360

    return angle

def classify_pose(landmarks):
    t_pose = False
    march_pose = False
    lea = calculate_angle(landmarks[11], landmarks[13], landmarks[15])
    rea = calculate_angle(landmarks[16], landmarks[14], landmarks[12])
    lsa = calculate_angle(landmarks[13], landmarks[11], landmarks[23])
    rsa = calculate_angle(landmarks[24], landmarks[12], landmarks[14])
    bal = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
    sar = calculate_angle(landmarks[12], landmarks[24], landmarks[26])
    bar = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
    sal = calculate_angle(landmarks[11], landmarks[23], landmarks[25])


    if (165 < lea <= 200) and (165 < rea <= 200) and (75 < rsa <= 115) and (75 < lsa <= 115):
        t_pose = True
    if ((80<=bal<=100) and (170<=sar<=190))  or ((80<=bar<=100) and (170<=sal<=190)):
        march_pose = True
    if(t_pose):
        pose_r = 1
    elif(march_pose):
        pose_r = 2
    else:
        pose_r = 0

    return pose_r

def process_video():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(0)  # 0 represents the default webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        landmarks = []

        if results.pose_landmarks:
            landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0], lm.z * frame.shape[1]) for lm in results.pose_landmarks.landmark]

            pose_r = classify_pose(landmarks)

            if (pose_r==1):
                result_text = "Pose:T-Pose"
            elif(pose_r==2):
                result_text = "Pose: March-pose"
            else:
                result_text = "Pose: Unknown"
            cv2.putText(frame, result_text, (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            drawing(frame, landmarks)

        cv2.imshow("Webcam Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the video stream
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()
