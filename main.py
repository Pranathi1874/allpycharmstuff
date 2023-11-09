import math
import mediapipe as m
import cv2


def detectPose(image, pose, display=True):
    output_image = image.copy()

    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(imageRGB)

    height, width, _ = image.shape

    landmarks = []

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)

        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                              (landmark.z * width)))

    if display:
        cv2.imshow("Original Image", image)
        cv2.imshow("Output Image", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        return output_image, landmarks



def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    if angle < 0:
        angle += 360

    return angle


def classifyPose(landmarks, output_image, display=False):
    T_Pose= False
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    if left_elbow_angle > 150 and left_elbow_angle <=180 and right_elbow_angle > 150 and right_elbow_angle <=180:

        if left_shoulder_angle > 65 and left_shoulder_angle <=90 and right_shoulder_angle > 65 and right_shoulder_angle <= 90:
            T_Pose = True
            while T_pose:
                errors = CalcError(left_elbow_angle, right_elbow_angle, right_shoulder_angle, left_shoulder_angle)

def CalcError(left_elbow_angle, right_elbow_angle, right_shoulder_angle, left_shoulder_angle):
    ideal_left_elbow_angle = 180
    ideal_right_elbow_angle = 180
    ideal_right_shoulder_angle = 90
    ideal_left_shoulder_angle = 90
    e1 = abs(ideal_left_shoulder_angle-left_shoulder_angle)
    e2 = abs(ideal_right_shoulder_angle-)
