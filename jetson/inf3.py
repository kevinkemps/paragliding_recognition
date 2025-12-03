from inference import get_model
import supervision as sv
import cv2
import time
# define the image url to use for inference
image_file = "./paraglider_recognition-8/test/images/P_20231121_114926_jpg.rf.193318024b03b5a48f7ca5bb7c37ae7d.jpg"
image = cv2.imread(image_file)

# load a pre-trained yolov8n model
model = get_model(model_id="paraglider_recognition/8", api_key="YhsWQtGvkDwCH0br9fSA")
time_list = []
# run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
for i in range(10):
    start = time.time()
    results = model.infer(image)[0]
    end = time.time()
    time_list.append(end - start)
average = sum(time_list) / len(time_list)
print(f"Result: {results}")
print(f"Test took {end - start} seconds")
print(f"test took an average of {average} seconds")
