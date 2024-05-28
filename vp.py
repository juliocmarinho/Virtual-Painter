import cv2
import mediapipe as mp
import os
import numpy as np
import time

#Function that executes the code.
def video_capture():
    #Define which camera to use and set dimensions and fps for video capture.
    video = cv2.VideoCapture(0)
    video.set(3, 1280)
    video.set(4, 720)
    video.set(cv2.CAP_PROP_FPS, 40)

    #Define initial points for drawing and the thickness of the brush and eraser.
    xp, yp = 0, 0
    brushThickness = 20
    eraserThickness = 50

    #Initialize variables for hand recognition in the video.
    hand = mp.solutions.hands
    Hand = hand.Hands(max_num_hands=1)
    mpDraw = mp.solutions.drawing_utils
    
    #Define colors and create an all-white window with zeros.
    drawColor = (255, 0, 255)
    imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    #Define the folder that contains images for the header.
    folderPath = "Header"
    myList = os.listdir(folderPath)
    overlayList = []
    for imPath in myList:
        image = cv2.imread(f'{folderPath}/{imPath}')
        overlayList.append(image)

    #Resize the header images to 1280x125.
    overlayList = [cv2.resize(img, (1280, 125)) for img in overlayList]

    header = overlayList[0]

    prev_time = 0

    #Main loop.
    while True:
        #Start video capture, flip the camera, and show fps.
        check, img = video.read()
        img = cv2.flip(img, 1)
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(img, f"FPS: {int(fps)}", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        #If there is an issue with the camera.
        if not check:
            print("Failed to capture the camera")
        #Convert img to RGB, the format that mediapipe accepts.
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        #Find hands and get landmarks.
        results = Hand.process(imgRGB)
        handsPoints = results.multi_hand_landmarks
        handedness = results.multi_handedness
        h, w, _ = img.shape
        pontos = []

        #If a hand is found:
        if handsPoints:
            for i, points in enumerate(handsPoints):
                #Identify which hand is in the camera (left or right hand).
                hand_label = handedness[i].classification[0].label
                #Draw connections on the hand.
                mpDraw.draw_landmarks(img, points, hand.HAND_CONNECTIONS)
                for id, cord in enumerate(points.landmark):
                    #Get positions of each point on the hand.
                    cx, cy = int(cord.x * w), int(cord.y * h)
                    pontos.append((id, cx, cy))
                    
                xp, yp = 0, 0
                
                #Select points for the index and middle fingers.
                x1, y1 = pontos[8][1:]
                x2, y2 = pontos[12][1:]

                #Determine which hand is in use and select the mode (Only index finger up: Drawing mode. Index and middle fingers up: Selection mode.)
                if hand_label == "Right":
                    selection_condition = pontos[8][1] < pontos[6][1] and pontos[12][1] > pontos[10][1] and pontos[16][1] < pontos[14][1] and pontos[20][1] < pontos[18][1]
                    drawing_condition = pontos[8][1] > pontos[6][1] and pontos[12][1] < pontos[10][1] and pontos[16][1] < pontos[14][1] and pontos[20][1] < pontos[18][1]
                else: # Left hand
                    selection_condition = pontos[8][1] > pontos[6][1] and pontos[12][1] < pontos[10][1] and pontos[16][1] > pontos[14][1] and pontos[20][1] > pontos[18][1]
                    drawing_condition = pontos[8][1] < pontos[6][1] and pontos[12][1] > pontos[10][1] and pontos[16][1] > pontos[14][1] and pontos[20][1] > pontos[18][1]

                if selection_condition:
                    xp, yp = 0, 0
                    #print("Selection Mode")
                    #Determine the position of the fingers and select the correct color.
                    if y1 < 125:
                        if 250 < x1 < 450:
                            header = overlayList[1]
                            drawColor = (255, 0, 255)
                        elif 550 < x1 < 750:
                            header = overlayList[2]
                            drawColor = (255, 0, 0)
                        elif 800 < x1 < 950:
                            header = overlayList[0]
                            drawColor = (0, 255, 0)
                        elif 1050 < x1 < 1200:
                            header = overlayList[3]
                            drawColor = (0, 0, 0)
                    # Draw a rectangle connecting both fingers.
                    cv2.rectangle(img, (x1, y1-15), (x2, y2+15), drawColor, cv2.FILLED)

                if drawing_condition:
                    # Draw a circle on the index finger.
                    cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
                    #print("Drawing Mode")
                    
                    #Ensure the line starts at the correct position when switching to drawing mode.
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1

                    #Check if the eraser is in use.
                    if drawColor == (0, 0, 0):
                        cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                        cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)

                    #Draw on the window.
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                    
                    #Keep the line drawn on the window following the index finger.
                    xp, yp = x1, y1

        #Configurations to allow drawing in the window. Converts imgCanvas (white) to gray.
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        #Convert the image to a reverse binary image.
        _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
        #Convert the binary image to BGR format.
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        #Keep the original image where it is not drawn.
        img = cv2.bitwise_and(img, imgInv)
        #Add the drawing to the original image.
        img = cv2.bitwise_or(img, imgCanvas)

        #Set the header and show the image.
        img[0:125, 0:1280] = header
        cv2.imshow("Image", img)
        #When 'q' is pressed, stop the program.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#Call the function to capture video.
def main():
    video_capture()

#Start the program.
if __name__ == '__main__':
    main()
