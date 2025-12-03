import inference
model = inference.load_roboflow_model("paraglider_recognition/8", api_key="YhsWQtGvkDwCH0br9fSA")
results = model.infer(image="./paraglider_recognition-8/test/images/P_20231121_114926_jpg.rf.193318024b03b5a48f7ca5bb7c37ae7d.jpg")
