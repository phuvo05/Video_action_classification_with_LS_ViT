import os
import xml.etree.ElementTree as ET
from utils import extract_data_from_xml, convert_to_YOLO_format, save_yolo_data
from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# BASE_DIR = D:/AIO/Video_action_classification_with_LS_ViT

dataset_dir = os.path.join(BASE_DIR, "datasets", "SceneTrialTrain")
words_xml_path = os.path.join(dataset_dir, "words.xml")

print(words_xml_path)


image_paths, image_sizes, image_labels, bounding_boxes = extract_data_from_xml(
    words_xml_path
)

yolo_data = convert_to_YOLO_format(image_paths, image_sizes, bounding_boxes)
print(f"Number of YOLO data: {len(yolo_data)}")

train_yolo_data,val_yolo_data = train_test_split(
    yolo_data,
    test_size=0.2,
    random_state=42
)
# test_yolo_data = train_test_split(val_yolo_data, test_size=0.5, random_state=42)

# print(len(train_yolo_data))
# print(len(val_yolo_data))
save_yolo_data_dir = os.path.join(BASE_DIR, "yolo_data")
save_yolo_data(train_yolo_data, "train", save_yolo_data_dir, dataset_dir)
save_yolo_data(val_yolo_data, "val", save_yolo_data_dir, dataset_dir)