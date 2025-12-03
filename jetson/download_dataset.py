from roboflow import Roboflow
rf = Roboflow(api_key="YhsWQtGvkDwCH0br9fSA")
project = rf.workspace("initialtrial").project("paraglider_recognition")
dataset = project.version(8).download("yolov8")
