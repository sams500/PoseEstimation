import cv2
import mediapipe as mp
import time


def init():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils
    previousTime = 0

    while True:
        success, img = cap.read()
        imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imageRGB)

        landMarks = results.pose_landmarks
        if landMarks:
            for id, lm in enumerate(landMarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id in [0, 11, 12, 23, 24, 15, 16]:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
            mpDraw.draw_landmarks(img, landMarks, mpPose.POSE_CONNECTIONS)

        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime
        cv2.putText(img, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 250), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    init()
