import numpy as np
import cv2
import imutils

NMS_THRESHOLD = 0.3
MIN_CONFIDENCE = 0.2

def pedestrian_detection(image, model, layer_name, personidz=0):
    (H, W) = image.shape[:2]
    results = []

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    model.setInput(blob)
    layerOutputs = model.forward(layer_name)

    boxes = []
    centroids = []
    confidences = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID == personidz and confidence > MIN_CONFIDENCE:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)
    
    if len(idzs) > 0:
        for i in idzs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            res = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(res)
            
    return results

labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

weights_path = "yolov4-tiny.weights"
config_path = "yolov4-tiny.cfg"

model = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# Uncomment the following lines if you have CUDA support
# model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer_name = model.getLayerNames()
layer_name = [layer_name[i - 1] for i in model.getUnconnectedOutLayers()]

# Load the image
image_path = "DJI_0858_cut.JPG"
image = cv2.imread(image_path)
if image is None:
    print(f"Error loading image: {image_path}")
else:
    image = imutils.resize(image, width=700)
    results = pedestrian_detection(image, model, layer_name, personidz=LABELS.index("person"))

    for res in results:
        # Count the number of pedestrians
        pedestrian_count = len(results)
        cv2.rectangle(image, (res[1][0], res[1][1]), (res[1][2], res[1][3]), (0, 255, 0), 2)

    # Display the number of pedestrians
    cv2.putText(image, f"Pedestrians: {pedestrian_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # Save the output image
    output_path = "output_image.jpg"
    cv2.imwrite(output_path, image)
    cv2.imshow("Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
