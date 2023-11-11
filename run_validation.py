from model import validation

from ultralytics import YOLO

if __name__ == '__main__':
    custom_model = YOLO('model/trained_model.pt')
    validation.run_validation(custom_model)
