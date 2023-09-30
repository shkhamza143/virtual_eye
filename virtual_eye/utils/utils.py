import math
import cv2
import os
import json
from skimage import exposure
from skimage.color import rgb2gray
from skimage import img_as_ubyte, img_as_float

import numpy as np


def get_image_mask(root, file):
    image = cv2.imread(os.path.join(root, file))
    json_path = os.path.join(root, file).replace('.tif', '.json')
    f = open(json_path)
    data = json.load(f)
    white_contour = []
    iris_contour = []
    black_contour = []
    for i in data['shapes']:
        if i['label'] == 'white':
            white_contour = i['points']
        elif i['label'] == 'iris':
            iris_contour = i['points']
        elif i['label'] == 'black':
            black_contour = i['points']
    x1 = int(min([inner[0] for inner in iris_contour]))
    y1 = int(min([inner[1] for inner in iris_contour]))
    x2 = int(max([inner[0] for inner in iris_contour]))
    y2 = int(max([inner[1] for inner in iris_contour]))
    bbox_iris = (x1, y1, x2, y2)

    x1 = int(min([inner[0] for inner in black_contour]))
    y1 = int(min([inner[1] for inner in black_contour]))
    x2 = int(max([inner[0] for inner in black_contour]))
    y2 = int(max([inner[1] for inner in black_contour]))

    bbox_black = (x1, y1, x2, y2)

    # print(white_contour)
    # for inner in white_contour:
    #     for i in inner:
    #         print(i)
    white_contour = np.array([[int(inner[0]), int(inner[1])] for inner in white_contour], np.int32).reshape(
        (-1, 1, 2))
    iris_contour = np.array([[int(inner[0]), int(inner[1])] for inner in iris_contour], np.int32).reshape(
        (-1, 1, 2))
    black_contour = np.array([[int(inner[0]), int(inner[1])] for inner in black_contour], np.int32).reshape(
        (-1, 1, 2))

    mask = np.zeros_like(image)
    # white_mask = mask.copy()
    iris_mask = mask.copy()
    black_mask = mask.copy()

    # white_mask = cv2.fillPoly(white_mask, [white_contour], (255, 255, 255))
    # white_mask = cv2.fillPoly(white_mask, [iris_contour], (0, 0, 0))

    iris_mask = cv2.fillPoly(iris_mask, [iris_contour], (255, 255, 255))
    # iris_mask = cv2.fillPoly(iris_mask, [black_contour], (0, 0, 0))

    black_mask = cv2.fillPoly(black_mask, [black_contour], (255, 255, 255))

    # cv2.waitKey(0)

    # white_image = mask.copy()
    iris_image = mask.copy()
    # black_image = mask.copy()

    # white_image[white_mask == 255] = image[white_mask == 255]
    iris_image[iris_mask == 255] = image[iris_mask == 255]
    # black_image[black_mask == 255] = image[black_mask == 255]
    # return ori_image, white_image, iris_image, black_image
    return iris_image, iris_mask, black_mask, bbox_iris, bbox_black


def align_images(imgRef, imgTest):
    imgTest_grey = cv2.cvtColor(imgTest, cv2.COLOR_BGR2GRAY)
    imgRef_grey = cv2.cvtColor(imgRef, cv2.COLOR_BGR2GRAY)
    height, width = imgRef_grey.shape

    # Configure ORB feature detector Algorithm with 1000 features.
    orb_detector = cv2.ORB_create(1000)

    # Extract key points and descriptors for both images
    keyPoint1, des1 = orb_detector.detectAndCompute(imgTest_grey, None)
    keyPoint2, des2 = orb_detector.detectAndCompute(imgRef_grey, None)

    # Display keypoints for reference image in green color
    imgKp_Ref = cv2.drawKeypoints(imgRef, keyPoint1, 0, (0, 250, 250), None)
    imgKp_Ref = cv2.resize(imgKp_Ref, (640, 480))

    cv2.imshow('Key Points', imgKp_Ref)
    cv2.waitKey(0)

    # Match features between two images using Brute Force matcher with Hamming distance
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the two sets of descriptors.
    matches = matcher.match(des1, des2)

    # Sort matches on the basis of their Hamming distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches) * 0.9)]
    no_of_matches = len(matches)

    # Display only 100 best matches {good[:100}
    imgMatch = cv2.drawMatches(imgTest, keyPoint2, imgRef, keyPoint1, matches[:100], None, flags=2)
    imgMatch = cv2.resize(imgMatch, (640, 480))

    cv2.imshow('Image Match', imgMatch)
    cv2.waitKey(0)

    # Define 2x2 empty matrices
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    # Storing values to the matrices
    for i in range(len(matches)):
        p1[i, :] = keyPoint1[matches[i].queryIdx].pt
        p2[i, :] = keyPoint2[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # Use homography matrix to transform the unaligned image wrt the reference image.
    aligned_img = cv2.warpPerspective(imgTest, homography, (width, height))
    return aligned_img


def apply_inpaint(source_image):
    source_image_mask = np.zeros_like(source_image)
    source_image_mask[source_image > 200] = 255
    source_image_mask = cv2.cvtColor(source_image_mask, cv2.COLOR_BGR2GRAY)
    source_image = cv2.inpaint(source_image, source_image_mask, 3, cv2.INPAINT_NS)
    return source_image


def shift_image(source_points, destination_image, destination_points):
    # Define the translation matrix to shift the image to the desired points
    dx = source_points[0] - destination_points[0]  # Shift in the x-direction
    dy = source_points[1] - destination_points[1]  # Shift in the y-direction
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted_image = cv2.warpAffine(destination_image, M, (destination_image.shape[1], destination_image.shape[0]))
    return shifted_image


def auto_blend(image1, image2, mask):
    blended = cv2.addWeighted(image1, 0.8, image2, 0.2, 0.0)
    return blended

    # # Convert images to float32 for blending
    # image1 = image1.astype(np.float32)
    # image2 = image2.astype(np.float32)
    #
    # # Normalize the mask to range [0, 1]
    # mask = mask.astype(np.float32) / 255.0
    #
    # # Multiply the images by the mask
    # blended = image1 * (mask - 1.0) + image2 * mask
    #
    # return blended.astype(np.uint8)


def adjust_exposure(image, exposure_factor=0.8):
    # Clip the pixel values to ensure they stay in the valid range [0, 255]
    adjusted_image = np.clip(image * exposure_factor, 0, 255).astype(np.uint8)
    return adjusted_image


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharp = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    sharp = np.maximum(sharp, np.zeros(sharp.shape))
    sharp = np.minimum(sharp, 255 * np.ones(sharp.shape))
    sharp = sharp.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharp, image, where=low_contrast_mask)
    return sharp


def post_process(image):
    print("Post Processing ...")
    # Apply Temperature
    image[:, :, 0] = np.clip(image[:, :, 0] + 0.98, 0, 255)  # Increase red channel
    image[:, :, 1] = np.clip(image[:, :, 1] + 0.98, 0, 255)  # Increase green channel

    # Apply Tint
    image[:, :, 0] = np.clip(image[:, :, 0] * 1.02, 0, 255)  # Scale down red channel
    image[:, :, 2] = np.clip(image[:, :, 2] * 1.02, 0, 255)  # Scale

    # Apply Contrast
    image = np.clip(image * 0.95, 0, 255).astype(np.uint8)

    # Apply Highlight
    image = np.clip(image * 1.1, 0, 255).astype(np.uint8)

    # Apply Shadow
    image = np.clip(image * 1.05, 0, 255).astype(np.uint8)

    # Apply the black and white adjustments
    image = np.clip(image * 1.1, 0, 255).astype(np.uint8)
    image = np.clip(image * 1.05, 0, 255).astype(np.uint8)

    # Apply texture adjustment
    image = np.clip(image * 0.84, 0, 255).astype(np.uint8)

    # Apply clarity adjustment
    image = cv2.detailEnhance(image, sigma_s=10, sigma_r=0.15 * 0.84)

    # Convert to HSV color space for vibrance and saturation adjustments
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Apply vibrance adjustment
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * 0.99, 0, 255).astype(np.uint8)

    # Apply saturation adjustment
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * 0.98, 0, 255).astype(np.uint8)

    # Convert back to BGR color space
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # Apply the dehaze effect
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Enhance local contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # Combine channels back into LAB image
    lab_image = cv2.merge((cl, a_channel, b_channel))

    # Convert back to BGR color space
    dehazed_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

    # Apply Gaussian blur for dehazing effect
    dehazed_image = cv2.GaussianBlur(dehazed_image, (5, 5), 0)

    # Apply the scaling factor
    image = np.clip(dehazed_image * 0.85, 0, 255).astype(np.uint8)
    return image


def reduced_points_to_360(original_points):
    if len(original_points) != 360:
        indices = np.linspace(0, len(original_points) - 1, 360, dtype=np.int32)
        updated_points = original_points[indices]
        return updated_points
    else:
        return original_points


# def generate_circular_points(radius, center):
#     points = []
#     for i in range(360):
#         angle = 2 * np.pi * i / 360
#         x = int(center[0] + (radius * np.cos(angle)))
#         y = int(center[1] + (radius * np.sin(angle)))
#         points.append((x, y))
#     return np.array(points, dtype=np.int32)


def find_angle(point, center):
    try:
        x, y = point
        x0, y0 = center
        # prependicular = math.sqrt((y - y0)**2)
        # base = math.sqrt((x - x0)**2)
        # Calculate the angle in radians
        angle = (math.atan2((y - y0),(x - x0)))*180/np.pi
        return angle

    except:
        return 90


def generate_circular_points(original_points, radius, center):
    points = []
    for i in range(360):
        angle = 2 * np.pi * i / 360
        x = int(center[0] + (radius * np.cos(angle)))
        y = int(center[1] + (radius * np.sin(angle)))
        points.append((x, y))
    return np.array(points, dtype=np.int32)


def find_boundary_points(mask):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.Canny(mask, 50, 150)  # You can adjust the threshold values as needed
    # Find contours of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract boundary points from the largest contour
    # boundary_points = []
    for c in contours:
        return c.squeeze()
    return []


def warp_image_points(image, source_points, target_points):
    print()
    # Define source and target points (4 points each)
    source_points = np.array(source_points.tolist(), dtype=np.float32)
    target_points = np.array(target_points.tolist(), dtype=np.float32)
    # Calculate the perspective transformation matrix
    M = cv2.getPerspectiveTransform(source_points, target_points)

    # Apply the perspective transformation to the image
    warped_image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))

    return warped_image


def warp_to_circular(image, center, radius):
    height, width = image.shape[:2]
    warped_image = np.zeros((radius * 2, radius * 2, 3), dtype=np.int32)

    for i in range(warped_image.shape[0]):
        for j in range(warped_image.shape[1]):
            x = j - radius
            y = i - radius
            theta = np.arctan2(y, x)
            r = np.sqrt(x ** 2 + y ** 2)
            if 0 <= r <= radius and 0 <= theta <= 2 * np.pi:
                src_x = int(center[0] + r * np.cos(theta))
                src_y = int(center[1] + r * np.sin(theta))
                if 0 <= src_x < width and 0 <= src_y < height:
                    warped_image[i, j] = image[src_y, src_x]

    return warped_image
