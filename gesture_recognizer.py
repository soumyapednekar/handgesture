import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from matplotlib import pyplot as plt
from mediapipe.framework.formats import landmark_pb2
import cv2
import math

import matplotlib
matplotlib.use('Agg')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

DESIRED_HEIGHT = 500
DESIRED_WIDTH = 500

def resize_and_show(image):
    if image is not None:  
        h, w = image.shape[:2]
        if h < w:
            img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
        else:
            img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    else:
        print("Image not loaded correctly.")

def display_one_image(image, title, subplot, titlesize=16):
    """Displays one image along with the predicted category name and score."""
    plt.subplot(*subplot)
    plt.imshow(image)
    if len(title) > 0:
        plt.title(title, fontsize=int(titlesize), color='black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
    return (subplot[0], subplot[1], subplot[2]+1)

def display_batch_of_images_with_gestures_and_hand_landmarks(images, results):
    # Images and labels.
    images = [image.numpy_view() for image in images]
    gestures = [top_gesture for (top_gesture, _) in results]
    multi_hand_landmarks_list = [multi_hand_landmarks for (_, multi_hand_landmarks) in results]

    #this will drop data that does not fit into square or square-ish rectangle.
    rows = int(math.sqrt(len(images)))
    cols = len(images) // rows

    # Size and spacing.
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot=(rows,cols, 1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))
    else:
        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))

    for i, (image, top_gesture) in enumerate(zip(images[:rows*cols], gestures[:rows*cols])):
        if top_gesture is not None:
            title = f"{top_gesture.category_name} ({top_gesture.score:.2f})"
        else:
            title = "No gesture detected"
        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols) * 40 + 3
        annotated_image = image.copy()

        if multi_hand_landmarks_list[i] is not None:
            for hand_landmarks in multi_hand_landmarks_list[i]:
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
                ])

                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks_proto,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        subplot = display_one_image(annotated_image, title, subplot, titlesize=dynamic_titlesize)

    # Layout.
    plt.tight_layout()
    plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()

def recognize_gestures_and_display(image_paths, output_path):
    base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)

    images = []
    results = []
    for image_path in image_paths:
        image = mp.Image.create_from_file(image_path)
        recognition_result = recognizer.recognize(image)
        images.append(image)
        if recognition_result.gestures and recognition_result.hand_landmarks:
            top_gesture = recognition_result.gestures[0][0]
            hand_landmarks = recognition_result.hand_landmarks
            results.append((top_gesture, hand_landmarks))
        else:
            results.append((None, None))

    display_batch_of_images_with_gestures_and_hand_landmarks(images, results)
    plt.savefig(output_path)
