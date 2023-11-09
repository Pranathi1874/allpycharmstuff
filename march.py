import cv2
import mediapipe as mp
import math

def drawing(img, landmarks):
    for landmark in landmarks:
        x, y, _ = landmark
    cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
def calculate_angle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    if angle < 0:
        angle += 360

    return angle

def classify_pose(landmarks):
    march_pose = False
    bal = calculate_angle(landmarks[23], landmarks[25], landmarks[27]) #bentangle
    sar = calculate_angle(landmarks[12], landmarks[24], landmarks[26])
    bar = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
    sal = calculate_angle(landmarks[11], landmarks[23], landmarks[25])
    '''lea = calculate_angle(landmarks[11], landmarks[13], landmarks[15])
    rea = calculate_angle(landmarks[16], landmarks[14], landmarks[12])
    lsa = calculate_angle(landmarks[13], landmarks[11], landmarks[23])
    rsa = calculate_angle(landmarks[24], landmarks[12], landmarks[14])

    if (165 < lea <= 195) and (165 < rea <= 195) and (75 < rsa <= 105) and (75 < lsa <= 105):'''
    if ((80<=bal<=100) and (170<=sar<=190))  or ((80<=bar<=100) and (170<=sal<=190)):
        march_pose = True

    return march_pose

fixed_bal = fixed_bar = 90
fixed_sal = fixed_sar = 180
def process_video():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        landmarks = []

        if results.pose_landmarks:
            landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0], lm.z * frame.shape[1]) for lm in results.pose_landmarks.landmark]

            march_pose = classify_pose(landmarks)

            if march_pose:
                result_text = "march-Pose: True"
            else:
                result_text = "march-Pose: False"
            bal = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
            angle_diff1 = fixed_bal-bal
            sar = calculate_angle(landmarks[12], landmarks[24], landmarks[26])
            angle_diff2 = fixed_sar-sar
            bar = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
            angle_diff3 = fixed_bar - bar
            sal = calculate_angle(landmarks[11], landmarks[23], landmarks[25])
            angle_diff4 = fixed_sal - sal
            total_error = (abs(angle_diff1) + abs(angle_diff2) + abs(angle_diff3) + abs(angle_diff4))/4
            angle_texta = f"Total Angle Difference: {total_error:.2f} degrees"
            cv2.putText(frame, angle_texta, (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, result_text, (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            drawing(frame, landmarks)

        cv2.imshow("Webcam Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the video stream
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()
