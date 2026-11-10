import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon
from ultralytics.utils.plotting import colors

def visualize_image_annotations_yolo_format(image_path, txt_path, output_path, label_map):

    img = np.array(Image.open(image_path))
    img_height, img_width = img.shape[:2]
    annotations = []
    with open(txt_path) as folder:
        for line in folder:
            class_id, x_center, y_center, width, height = map(float, line.split()[:5])
            x = (x_center - width / 2) * img_width
            y = (y_center - height / 2) * img_height
            w = width * img_width
            h = height * img_height
            annotations.append((x, y, w, h, int(class_id)))
    fig, ax = plt.subplots(1)  # Plot the image and annotations
    for x, y, w, h, label in annotations:
        color = tuple(c / 255 for c in colors(label, True))  # Get and normalize the RGB color
        rect = Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor="none")  # Create a rectangle
        ax.add_patch(rect)
        luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]  # Formula for luminance
        ax.text(x, y - 5, label_map[label], color="white" if luminance < 0.5 else "black", backgroundcolor=color)
    ax.imshow(img)
    # save to file
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

def visualize_image_annotations_nomnaocr_format(image_path, txt_path, output_path, label_map):
    img = np.array(Image.open(image_path))
    # img_height, img_width = img.shape[:2]
    annotations = []
    with open(txt_path) as folder:
        for line in folder:
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, line.split(",")[:8])
            annotations.append((x1, y1, x2, y2, x3, y3, x4, y4, 0))
    fig, ax = plt.subplots(1)  
    for x1, y1, x2, y2, x3, y3, x4, y4, _ in annotations:
        color = tuple(c / 255 for c in colors(0, True))  
        pts = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        poly = Polygon(pts, closed=True, linewidth=1, edgecolor=color, facecolor="none")
        ax.add_patch(poly)

        # luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]  # Formula for luminance
        # ax.text(x, y - 5, label_map[label], color="white" if luminance < 0.5 else "black", backgroundcolor=color)
    ax.imshow(img)
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

if __name__ == "__main__":
    
    # visualize_image_annotations_yolo_format(
    # image_path="/home/projectg7/workspace/nom-ocr-project/datasets/nom-detection-testset/images/DVSKTT_ban_tuc_XVII_53b_jpg.rf.67a6bb4e370470b81a48df3ce20466ff.jpg",
    # txt_path="/home/projectg7/workspace/nom-ocr-project/datasets/nom-detection-testset/labels/DVSKTT_ban_tuc_XVII_53b_jpg.rf.67a6bb4e370470b81a48df3ce20466ff.txt",
    # output_path="visualize/check.jpg",
    # label_map={0: ""},
    # )

    visualize_image_annotations_nomnaocr_format(
    image_path="/home/projectg7/workspace/nom-ocr-project/datasets/nomnaocr/images/DVSKTT-3 Ban ky toan thu/DVSKTT_ban_toan_I_16a.jpg",
    txt_path="/home/projectg7/workspace/nom-ocr-project/datasets/nomnaocr/line_labels/DVSKTT-3 Ban ky toan thu/DVSKTT_ban_toan_I_16a.txt",
    output_path="visualize/line_label3.jpg",
    label_map={0: ""},
    )

    # visualize_image_annotations_yolo_format(
    # image_path="/home/projectg7/workspace/nom-ocr-project/projectg7_finetune/datasets/nomnaocr/Tale of Kieu 1871/images/page050.jpg",
    # txt_path="/home/projectg7/workspace/nom-ocr-project/projectg7_finetune/datasets/nomnaocr/Tale of Kieu 1871/labels/page050.txt",
    # output_path="visualize/filter_label3.jpg",
    # label_map={0: ""},
    # )