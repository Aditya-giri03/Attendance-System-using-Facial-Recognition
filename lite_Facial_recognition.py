import numpy as np
import tensorflow as tf
import cv2
import pathlib
from datetime import datetime
import pandas as pd


def predict(interpreter, name):
    attendence = dict({})
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.allocate_tensors()

    # input details
    print(input_details)
    # output details
    print(output_details)

    cam = cv2.VideoCapture(0)

    model = cv2.CascadeClassifier(
        "./haarcascade_frontalface_alt.xml"
    )
    offset = 20
    # Read image from camera

    while True:
        success, img = cam.read()
        if not success:
            print("Cannot Read From Camera")
            return

        faces = model.detectMultiScale(img, 1.1, 6)
        i = 0
        for f in faces:
            i += 1
            x, y, w, h = f

            cropped_face = img[y - offset : y + h + offset, x - offset : x + w + offset]
            cropped_shape = cropped_face.shape
            # print(cropped_shape)
            if cropped_shape[0] > 100 and cropped_shape[1] > 100:
                cropped_face = cv2.resize(cropped_face, (100, 100))
                IMG_SIZE = 100

                # cropped_face = cropped_face.flatten().reshape(1, -1)
                print(cropped_face.shape)
                # Predict class
                gray = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
                face = np.array(gray).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

                face = face / 255.0
                face = face.astype(np.float32)
                print(face.shape)

                interpreter.set_tensor(input_details[0]["index"], face)

                # run the inference
                interpreter.invoke()

                # output_details[0]['index'] = the index which provides the input
                prediction = interpreter.get_tensor(output_details[0]["index"])

                # prediction = intrepreter.predict(face)
                # y_predict = model.predict(X_test)
                print(prediction)
                output = prediction.argmax(axis=1)
                print(output)
                output = int(output)
                print(output)
                print(name)
                print(type(name), type(output))
                if prediction[0][output] > 0.98:
                    if name[output] in attendence:
                        print("present")
                        print(attendence)
                    else:
                        attendence[name[output]] = datetime.now().strftime(
                            "%m/%d/%Y, %H:%M:%S"
                        )
                        print(attendence)

                    namePredicted = (
                        name[output]
                        + " "
                        + str(round(prediction[0][output] * 100, 2))
                        + "%"
                    )
                else:
                    namePredicted = "Unknown"

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    img,
                    namePredicted,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

            # cv2.imshow("Cropped" + str(i), cropped_face)

        cv2.imshow("Image Window", img)

        key = cv2.waitKey(10)
        if key == ord("q"):
            df = pd.DataFrame.from_dict(attendence, orient="index", columns=["Value"])
            df.reset_index(inplace=True)
            df.rename(columns={"index": "Attendence"}, inplace=True)

            # Save the DataFrame to a CSV file
            df.to_csv("./Attendence/attendence.csv", index=False, encoding="utf-8")
            break

    cam.release()
    cv2.destroyAllWindows()


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="./lite_facial_recognition_model2.tflite")

# Get input and output tensors.

name_map = np.load("Attendence/name_map.npy", allow_pickle=True)
name_map = name_map.item()
print('NAME MAP:: ',name_map, type(name_map))

predict(interpreter, name_map)


