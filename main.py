import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np 

def load_network():
    net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg")
    return net

def load_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def prepare_classes():
    classes=None
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

def prepare_img(image, net, output_layers):
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    return outs


st.title("Object Detection")
upload = st.file_uploader(label="Upload Image", type = ["jpg","png","jpeg"])
button = st.button("Analyse Image")

if button and upload:
    image_pil = Image.open(upload).convert('RGB')
    img = np.array(image_pil)[:,:,::-1].copy()
    height, width, channels = img.shape

    net = load_network()
    output_layers = load_output_layers(net)
    classes = prepare_classes()
    outs = prepare_img(img, net, output_layers)


    class_ids = []
    confidences = []
    boxes = []
    labels = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            labels.append(label)
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), font, 1, color, 2)
    fig = plt.figure(figsize=(12,12))
    plt.imshow(img)
    st.pyplot(fig, use_container_width=True)
    st.header("Objects Found")
    st.write(labels)