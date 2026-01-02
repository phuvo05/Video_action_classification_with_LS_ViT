import os
import xml.etree.ElementTree as ET
import shutil
def extract_data_from_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    
    image_paths = []
    image_size = []
    image_labels = []
    bounding_boxes = []


    for image in root:
        bbs_of_image = []
        labels_of_image = []

        for bbs in image.findall('taggedRectangles'):

            for bb in bbs:

                if not bb[0].text.isalnum():
                    continue
                if "é" in bb[0].text.lower() or "ñ" in bb[0].text.lower():
                    continue
                bbs_of_image.append([
                    float(bb.attrib['x']),
                    float(bb.attrib['y']),
                    float(bb.attrib['width']),
                    float(bb.attrib['height'])
                ])
                labels_of_image.append(bb[0].text.lower())

        image_paths.append(image[0].text.lower())
        image_size.append([
            int(image[1].attrib['x']),
            int(image[1].attrib['y'])            
        ])
        bounding_boxes.append(bbs_of_image)
        image_labels.append(labels_of_image)
    
    return image_paths, image_size, image_labels, bounding_boxes

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# BASE_DIR = D:/AIO/Video_action_classification_with_LS_ViT

dataset_dir = os.path.join(BASE_DIR, "datasets", "SceneTrialTrain")
words_xml_path = os.path.join(dataset_dir, "words.xml")

image_paths, image_sizes, image_labels, bounding_boxes = extract_data_from_xml(
    words_xml_path
)
# print(f"Total images: {len(image_paths)}")
# print(f"Example image path: {image_paths[0]}")
# print(f"Example image size: {image_sizes[0]}")
# print(f"Example image labels: {image_labels[0]}")
# print(f"Example bounding boxes: {bounding_boxes[0]}")



def convert_to_YOLO_format(image_paths, image_sizes, bounding_boxes):
    yolo_data = []
    for image_path, image_size, bbs in zip(image_paths, image_sizes, bounding_boxes):
        yolo_bbs_of_image = []
        img_width, img_height = image_size

        for bb in bbs:
            x,y,w,h = bb 
            x_center = (x + w / 2 ) / img_width
            y_center = (y + h / 2) / img_height
            width_norm = w / img_width
            height_norm = h / img_height

            yolo_bbs_of_image.append([0,x_center,y_center,width_norm,height_norm])

        yolo_data.append((image_path, yolo_bbs_of_image))
    return yolo_data

yolo_data = convert_to_YOLO_format(image_paths, image_sizes, bounding_boxes)
# print(f"Number of YOLO data: {len(yolo_data)}")
# print(f"Example YOLO data: {yolo_data[0]}")


def save_yolo_data(data, split, save_dir, dataset_dir):
    split_dir = os.path.join(save_dir, split)
    images_dir = os.path.join(split_dir, "images")
    labels_dir = os.path.join(split_dir, "labels")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for image_path, bbs in data:
        # Normalize the image path and create a clean filename
        image_name = image_path.replace("/", "_")
        
        # Source: original dataset location
        image_src = os.path.normpath(os.path.join(dataset_dir, image_path))
        
        # Destination: new YOLO directory
        image_dest = os.path.join(images_dir, image_name)

        shutil.copyfile(image_src, image_dest)

        label_name = image_name.replace(".jpg", ".txt")
        label_path = os.path.join(labels_dir, label_name)

        with open(label_path, "w") as f:
            for bb in bbs:
                f.write(f"{bb[0]} {bb[1]} {bb[2]} {bb[3]} {bb[4]}\n")

    print(f"Saved {len(data)} images and labels to {split_dir}")

# Example usage:
from sklearn.model_selection import train_test_split
save_dir = os.path.join(BASE_DIR, "datasets", "YOLO_SceneTrialTrain")
dataset_dir = os.path.join(BASE_DIR, "datasets", "SceneTrialTrain")
train_yolo_data,val_yolo_data = train_test_split(
    yolo_data,
    test_size=0.2,
    random_state=42
)

save_yolo_data_dir = os.path.join(BASE_DIR, "yolo_data")
save_yolo_data(train_yolo_data, "train", save_yolo_data_dir, dataset_dir)
save_yolo_data(val_yolo_data, "val", save_yolo_data_dir, dataset_dir)


