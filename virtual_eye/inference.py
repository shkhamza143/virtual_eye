import time
from utils.utils import *
from utils.warping import warp_points


def main(src, dst_files):
    start_time = time.time()
    name = src.split('.')[0]
    source_image, iris_mask, black_mask, bbox_iris, bbox_black = get_image_mask('annotations', src)
    for file in dst_files:
        target_image, _, _, _, bbox_black_target = get_image_mask('annotations', file)
        target_image = shift_image(bbox_black, target_image, bbox_black_target)
        source_image = auto_blend(source_image, target_image, iris_mask)

    source_image = adjust_exposure(source_image)
    source_image = apply_inpaint(source_image)
    x1, y1, x2, y2 = bbox_iris
    center = int((x1 + x2) / 2), int((y1 + y2) / 2)

    # Warp IRIS mask
    radius = int(min([abs(x2 - x1) / 2, (y2 - y1) / 2]))
    image_points = find_boundary_points(source_image)
    image_points = reduced_points_to_360(image_points)
    circular_points = generate_circular_points(image_points, radius, center)

    source_image = warp_points(source_image, image_points, circular_points)
    source_image = source_image[int(bbox_iris[1]): int(bbox_iris[3]), int(bbox_iris[0]): int(bbox_iris[2])]
    source_image = unsharp_mask(source_image, amount=3)
    source_image = post_process(source_image)
    cv2.imwrite(f'{name}_1_circle.tif', source_image)
    end_time = time.time()
    print(f"Total execution time is: {end_time - start_time}")


if __name__ == '__main__':
    source_file = 'annotation.tif'
    destination_files = ['1.tif', '2.tif', '3.tif', '4.tif', '5.tif']
    main(source_file, destination_files)
