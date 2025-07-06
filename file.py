import cv2
import mediapipe as mp
import math


def distance(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def slope(x1,y1,x2,y2):
    return ((y2-y1)/(x2-x1))


def calc(value):
    # Calculate arctangent in radians
    atan_radians = math.atan(value)
    # Convert radians to degrees
    return  math.degrees(atan_radians)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
velocity = 30
theta = 50
X_center = 320
y_center = 240
# z = 0
screen_width = 640
screen_height = 480
radius = 25
while cap.isOpened():
    # print(5)

    x_theta = math.cos(math.radians(theta))
    y_theta = math.sin(math.radians(theta))

    X_center = X_center + int(velocity * x_theta)
    y_center = y_center + int(velocity * y_theta)
    if X_center >screen_width-radius or X_center <radius:
        theta = 180-theta
    elif y_center <radius + 15 or y_center >screen_height-radius:
        theta = 360-theta

    ret,frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    # z = 0
    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


            if (distance(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * 640,hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * 480,X_center,y_center))<radius:
                val = slope(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * 640,
                            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * 480,
                            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * 640,
                            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * 480)
                if x_theta > y_theta:
                    theta = 180 - theta
                else:
                    theta = 360 - theta
            elif (distance(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * 640,hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * 480,X_center,y_center))<radius:

                print('collision-7')
                # val = slope(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * 640,hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * 480,hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * 640,hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * 480)
                # print(name)
                val = slope(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * 640,
                            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * 480,
                            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * 640,
                            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * 480)
                if x_theta > y_theta:
                    theta = 180 - theta
                else:
                    theta = 360 - theta

            elif (distance(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * 640,hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * 480,X_center,y_center))<radius:
                val = slope(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * 640,
                            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * 480,
                            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * 640,
                            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * 480)
                if x_theta > y_theta:
                    theta = 180 - theta
                else:
                    theta = 360 - theta
            elif (distance(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * 640,hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * 480,X_center,y_center))<radius:
                val = slope(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * 640,
                            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * 480,
                            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * 640,
                            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * 480)
                if x_theta > y_theta:
                    theta = 180 - theta
                else:
                    theta = 360 - theta
            elif (distance(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * 640,hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * 480,X_center,y_center))<radius:
                val = slope(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * 640,
                            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * 480,
                            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * 640,
                            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * 480)
                if x_theta > y_theta:
                    theta = 180 - theta
                else:
                    theta = 360 - theta
            elif (distance(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * 640,hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * 480,X_center,y_center))<radius:
                val = slope(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * 640,
                            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * 480,
                            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * 640,
                            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * 480)
                if x_theta > y_theta:
                    theta = 180 - theta
                else:
                    theta = 360 - theta
            elif (distance(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * 640,hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * 480,X_center,y_center))<radius:
                val = slope(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * 640,
                            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * 480,
                            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * 640,
                            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * 480)
                if x_theta > y_theta:
                    theta = 180 - theta
                else:
                    theta = 360 - theta
            elif (distance(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * 640,hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * 480,X_center,y_center))<radius:
                val = slope(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * 640,
                            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * 480,
                            hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * 640,
                            hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * 480)
                if x_theta > y_theta:
                    theta = 180 - theta
                else:
                    theta = 360 - theta
            elif (distance(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * 640,hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * 480,X_center,y_center))<radius:
                val = slope(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * 640,
                            hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * 480,
                            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * 640,
                            hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * 480)
                if x_theta > y_theta:
                    theta = 180 - theta
                else:
                    theta = 360 - theta
            elif (distance(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * 640,hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * 480,X_center,y_center))<radius:
                val = slope(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * 640,
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * 480,
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * 640,
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * 480)
                if x_theta > y_theta:
                    theta = 180 - theta
                else:
                    theta = 360 - theta
            elif (distance(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * 640,hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * 480,X_center,y_center))<radius:
                val = slope(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * 640,
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * 480,
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * 640,
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * 480)
                if x_theta > y_theta:
                    theta = 180 - theta
                else:
                    theta = 360 - theta
            elif (distance(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * 640,hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * 480,X_center,y_center))<radius:
                val = slope(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * 640,
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * 480,
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * 640,
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * 480)
                if x_theta > y_theta:
                    theta = 180 - theta
                else:
                    theta = 360 - theta
            elif (distance(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * 640,hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * 480,X_center,y_center))<radius + 40:
                val = slope(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * 640,
                hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * 480,
                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * 640,
                hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * 480)
                if x_theta > y_theta:
                    theta = 180 - theta
                else:
                    theta = 360 - theta
    frame = cv2.circle(frame,(X_center,y_center),radius,(255,0,0),cv2.FILLED)

    if ret:
            cv2.imshow('CAM',frame)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()