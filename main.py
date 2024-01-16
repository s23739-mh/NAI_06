"""
Autorzy:
Hołdakowski, Mikołaj (s23739)
Prętki, Mikołaj (s22982)
Baba Jaga Patrzy
Wymagany import: cv2
Należy zainstalować bibliotekę cv2 i uruchomić skrypt, w przypadku poprawnego działania programu, wyskoczyć powinno
okno z przechwytywanym obrazem z domyślnej dostępnej kamery. Na środku twarzy powinien pojawić się zielony kursor, zaś
w wypadku ruchu twarzy - czerwony prostokąt na twarzy. By opuścić program należy wcisnąć 'Q' bądź 'Esc'.
"""

import cv2
import time


def drawCrosshair(x: int, y: int, width: int, height: int, frame):
    """
    Function drawing crosshair on detected face
    :param x: x coordinate of face
    :param y: y coordinate of face
    :param width: width of face on a screen
    :param height: height of face on a screen
    :param frame: processed frame captured from video
    :return: none
    """
    xFace = int(x + width / 2)
    yFace = int(y + height / 2)
    markerSize = int((width + height) / 8)
    cv2.drawMarker(frame, (xFace, yFace), (0, 255, 0), markerType=cv2.MARKER_CROSS,
                   markerSize=markerSize, thickness=2)
    print(f"width: {width} height: {height} xFace: {xFace} yFace: {yFace}")


def drawRectangle(x: int, y: int, width: int, height: int, frame):
    """
    Function drawing a red rectangle on top of detected face.
    :param x: x coordinate of face
    :param y: y coordinate of face
    :param width: width of face on a screen
    :param height: height of face on a screen
    :param frame: processed frame captured from video
    :return: none
    """
    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 3)


def isMoving(previousPoint: (int, int), currentPoint: (int, int), width: int, height: int):
    """
    Function checks for movement of face displayed on camera. It takes into consideration previous frame, current frame
    and size of detected face. Then it calculates coordinates change of the detected face and checks if any of those are
    above acceptable change. During program implementation, it seemed okay to allow 5% change of width and height mean value.
    Program allows for slight movement because even when sitting completely still, the coordinates may change.
    :param previousPoint: x,y coordinates of a face during previous captured frame
    :param currentPoint: x,y coordinates of a face during current captured frame
    :param width: width of face on a screen
    :param height:height of face on a screen
    :return: boolean indicating if movement has been detected
    """
    x1, y1 = previousPoint
    x2, y2 = currentPoint
    acceptableMovementRatio = 0.05
    acceptableCordsChange = (width + height / 2) * acceptableMovementRatio
    print(f"acceptableCordsChange: {acceptableCordsChange}")
    if (abs(x1 - x2) > acceptableCordsChange or abs(y1 - y2) > acceptableCordsChange):
        return True
    return False


def renderCamera(frame):
    """
    Function showing user the camera.
    :param frame: processed frame captured from video
    :return: none
    """
    cv2.imshow("Movement Detector", frame)


def detectedMovementFunction(frame):
    """
    [Function not implemented into the program] In case of detected movement,
     draws the screen one last time and stops the program.
    :param frame: processed frame captured from video
    :return: none
    """
    renderCamera(frame)
    raise IOError('You moved! You are dead.')


def loadFaceDetector(path):
    """
    Function loading the face detector (Cascade Classifier) from a file.
    :param path: relative path leading to xml file
    :return: face_cascade - Cascade Classifier class created using data from file
    """
    face_cascade = cv2.CascadeClassifier(path)
    if face_cascade.empty():
        raise IOError('Unable to load')
    return face_cascade


def runCameraLoop(face_cascade, cap):
    """
    Loop responsible for proper video capture display. It contains logic of other methods responsible for drawing
    crosshair on a captured face, rectangle over captured face in case of detected movement. Method quits loop after
    pressing 'q' or 'Esc' buttons.
    :param face_cascade: Cascade Classifier class
    :param cap: Instance of started video capture
    :return: none
    """
    isFirstFrame = True
    previous_points: (int, int)

    while True:
        _, frame = cap.read()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        for (x, y, w, h) in face_rects:
            if not isFirstFrame:
                if isMoving(previous_points, (x, y), w, h):
                    drawRectangle(x, y, w, h, frame)
                    drawCrosshair(x, y, w, h, frame)
                    previous_points = (x, y)
                    # detectedMovementFunction(frame)
                else:
                    drawCrosshair(x, y, w, h, frame)
            else:
                isFirstFrame = False
                previous_points = (x, y)
                drawCrosshair(x, y, w, h, frame)

        renderCamera(frame)

        if (cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) == 27):
            return


def main():
    """
    Main function combining rest of functions
    :return: none
    """
    face_detector_path = 'face_detector.xml'

    face_cascade = loadFaceDetector(face_detector_path)
    cap = cv2.VideoCapture(0)

    runCameraLoop(face_cascade, cap)

    cap.release()
    cv2.destroyAllWindows()


# if __name__ == '__main__':
main()
