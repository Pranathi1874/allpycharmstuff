import cv2
import mediapipe as mp
import math

def drawing(img, landmarks):
    for landmark in landmarks:
        x, y, _ = landmark
    cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), 5)
def calculate_angle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    if angle < 0:
        angle += 360

    return angle

def classify_pose(landmarks):
    plank_pose = False
    lha = calculate_angle(landmarks[11], landmarks[23], landmarks[25])  # lefthipangle
    lka = calculate_angle(landmarks[23], landmarks[25], landmarks[27])  # leftkneeangle
    rha = calculate_angle(landmarks[12], landmarks[24], landmarks[26])
    rka = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
    rea = calculate_angle(landmarks[12], landmarks[14], landmarks[16])  # rightelbowangle
    lea = calculate_angle(landmarks[11], landmarks[13], landmarks[15])

    if (
            (155 < lha <= 215) and (155 < lka <= 215) and
            (155 < rha <= 195) and (155 < rka <= 215) and
            (65 < lea <= 115) and (65 < rea <= 115)
    ):
        plank_pose = True

    return plank_pose


fixed_rha_angle = fixed_lha_angle = fixed_rka_angle = fixed_lka_angle = 180
fixed_lea_angle = fixed_rea_angle = 90


def process_video(vidpath):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(vidpath)
    nw = 640
    nh = 480

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (nw, nh))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        landmarks = []

        if results.pose_landmarks:
            landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0], lm.z * frame.shape[1]) for lm in results.pose_landmarks.landmark]

            plank_pose = classify_pose(landmarks)

            if plank_pose:
                result_text = "plank-Pose: True"
            else:
                result_text = "plank-Pose: False"
            lha = calculate_angle(landmarks[11], landmarks[23], landmarks[25])
            angle_diff1 = fixed_rea_angle - lha
            lka = calculate_angle(landmarks[23], landmarks[25], landmarks[27])
            angle_diff2 = fixed_lka_angle - lka
            rha = calculate_angle(landmarks[12], landmarks[24], landmarks[26])
            angle_diff3 = fixed_rha_angle - rha
            rka = calculate_angle(landmarks[24], landmarks[26], landmarks[28])
            angle_diff4 = fixed_rka_angle - rka
            rea = calculate_angle(landmarks[12], landmarks[14], landmarks[16])
            angle_diff5 = fixed_rea_angle - rea
            lea = calculate_angle(landmarks[11], landmarks[13], landmarks[15])
            angle_diff6 = fixed_lea_angle - lea
            total_error = (abs(angle_diff1) + abs(angle_diff2) + abs(angle_diff3) + abs(angle_diff4) + abs(angle_diff5) + abs(angle_diff6))/4
            angle_texta = f"Total Angle Difference: {total_error:.2f} degrees"
            cv2.putText(frame, angle_texta, (70, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, result_text, (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            print(lha)

            drawing(frame, landmarks)

        cv2.imshow("Webcam Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the video stream
            breakq

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    vidpath = r"C:\Users\pranathi\Downloads\pexels-roman-odintsov-6152665 (2160p).mp4"
    process_video(vidpath)
