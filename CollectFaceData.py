import cv2
import numpy as np


def capture_face():
    cam = cv2.VideoCapture(0)

    # ask for name
    name = input("Enter Your Name : ")
    dataset_path = "./Image_data/"
    offset = 20
    facedata = []
    model = cv2.CascadeClassifier(
        "./haarcascade_frontalface_alt.xml"
    )

    # Read image from camera
    skip = 0
    while True:
        success, img = cam.read()
        if not success:
            print("Cannot Read From Camera")

        # store grayscale image
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = model.detectMultiScale(img, 1.3, 5)
        faces = sorted(faces, key=lambda f: f[2] * f[3])

        if len(faces) > 0:
            f = faces[-1]
            x, y, w, h = f
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cropped_face = img[y - offset : y + h + offset, x - offset : x + w + offset]
            crop_shape = cropped_face.shape
            if crop_shape[0] > 100 and crop_shape[1] > 100:
                cropped_face = cv2.resize(cropped_face, (100, 100))

                skip += 1
                if skip % 10 == 0:
                    gray = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
                    facedata.append(gray)
                    print("face captured ", len(facedata))
                if len(facedata) > 100:
                    break
            # cv2.imshow("Cropped", cropped_face)

        cv2.imshow("Image Window", img)

        key = cv2.waitKey(10)
        if key == ord("q"):
            break

    #  flattening

    # facedata = np.asarray(facedata)
    # print(facedata.shape)
    # m = facedata.shape[0]
    # facedata = facedata.reshape((m, -1))
    # print(facedata.shape)

    file = dataset_path + name + ".npy"
    np.save(file, facedata)

    cam.release()
    cv2.destroyAllWindows()


# capture_face()
