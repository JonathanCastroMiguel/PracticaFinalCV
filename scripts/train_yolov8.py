from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolov8m.yaml")  # build a new model from scratch
    # Use the model
    model.train(data = r"/Users/jon/Desktop/PracticaFinalCV/data/dataset.yaml",cfg=r"/Users/jon/Desktop/PracticaFinalCV/cfg/default.yaml")  # train the model
