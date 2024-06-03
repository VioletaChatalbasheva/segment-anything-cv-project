from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import os
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(DEVICE)


def read_image(image_name):
    img = cv2.imread(os.path.join(os.getcwd(), 'segment_anything', 'images', image_name))
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return image


def initialize_model():
    sam = sam_model_registry["vit_l"](checkpoint=os.path.join(os.getcwd(), 'segment_anything', 'ckpt', 'sam_vit_l_0b3195.pth')).to(DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator


def is_central(bbox, image_shape):
    img_center_x, img_center_y = image_shape[1] // 2, image_shape[0] // 2
    bbox_center_x = bbox[0] + bbox[2] / 2
    bbox_center_y = bbox[1] + bbox[3] / 2
    return abs(bbox_center_x - img_center_x) < image_shape[1] * 0.1 and abs(bbox_center_y - img_center_y) < image_shape[0] * 0.1


def extract_garment(mask_generator, image, image_name, count):
    masks = mask_generator.generate(image)
    # filtered_masks = [mask for mask in masks if 100000 < mask['area'] < 200000]
    filtered_masks = [mask for mask in masks if 90000 < mask['area'] < 200000]
    central_anns = [ann for ann in filtered_masks if is_central(ann['bbox'], image.shape)]
    # print(len(filtered_masks))

    if len(central_anns):
        garment_mask = central_anns[0]['segmentation']
        # sv.plot_image(image=garment_mask, size=(img.shape[0] // 100, img.shape[1] // 100))

        garment_image = np.uint8(garment_mask) * 255
        index = image_name.find('.')
        mask_name = image_name[:index] + '_mask' + image_name[index:]
        cv2.imwrite(os.path.join(os.getcwd(),  'segment_anything', 'images', 'masks', mask_name), garment_image)
    else:
        count += 1
        print(f'No garment found in {image_name}')
    return count


def automate_segmentation():
    mask_generator = initialize_model()
    count = 0
    for image_name in os.listdir(os.path.join(os.getcwd(), 'segment_anything', 'images')):
        if image_name.endswith('.jpg'):
            image = read_image(image_name)
            old_count = count
            count = extract_garment(mask_generator, image, image_name, count)
            if old_count != count:
                print(count)


def process_masks(masks, image):
    # get all masks area
    masks_areas = [mask['area'] for mask in masks]
    areas_sorted = sorted(masks_areas, reverse=True)
    print(areas_sorted)

    filtered_masks = [mask for mask in masks if 90000 < mask['area'] < 200000]
    central_anns = [ann for ann in filtered_masks if is_central(ann['bbox'], image.shape)]
    print([ann['area'] for ann in central_anns])

    plot_masks(masks)


def plot_masks(masks):
    masks_to_plot = [
        mask['segmentation']
        for mask
        in sorted(masks, key=lambda x: x['area'], reverse=True)
    ]
    print(len(masks_to_plot))

    sv.plot_images_grid(
        images=masks_to_plot,
        grid_size=(4, int(len(masks_to_plot) / 2)),
        size=(16, 16)
    )


def show_anns(anns):
    if len(anns) == 0:
        return

    # sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    print(sorted_anns)

    filtered_anns = [mask for mask in anns if 100000 < mask['area'] < 200000]

    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in filtered_anns:
        print(ann['area'])
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        np.dstack((img, m * 0.35))
        ax.imshow(np.dstack((img, m * 0.35)))

# Plot all masks on image
# plt.figure(figsize=(12, 9))
# plt.imshow(image)
# show_anns(masks)
# plt.axis('off')
# plt.savefig(os.path.join(os.getcwd(), 'segment_anything', 'images', '00006_00_garment.jpg'), bbox_inches='tight')


if __name__ == '__main__':
    # Test all images
    automate_segmentation()

    # Test 1 image
    # mask_generator = initialize_model()
    # image = read_image('00006_00.jpg')
    # masks = mask_generator.generate(image)
    # process_masks(masks)
