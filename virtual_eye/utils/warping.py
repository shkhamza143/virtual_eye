import math

import cv2
import numpy as np


def triangles(points):
    points = np.where(points, points, 1)
    subdiv = cv2.Subdiv2D((*points.min(0), *points.max(0)))
    for pt in points:
        subdiv.insert(tuple(map(int, pt)))
    for pts in subdiv.getTriangleList().reshape(-1, 3, 2):
        yield [np.where(np.all(points == pt, 1))[0][0] for pt in pts]


def crop(img, pts):
    x, y, w, h = cv2.boundingRect(pts)
    img_cropped = img[y: y + h, x: x + w]
    pts[:, 0] -= x
    pts[:, 1] -= y
    return img_cropped, pts


def warp_points(img1, pts1, pts2):
    img2 = np.zeros_like(img1)
    for indices in triangles(pts1):
        img1_cropped, triangle1 = crop(img1, pts1[indices])
        img2_cropped, triangle2 = crop(img2, pts2[indices])
        transform = cv2.getAffineTransform(np.float32(triangle1), np.float32(triangle2))
        img2_warped = cv2.warpAffine(img1_cropped, transform, img2_cropped.shape[:2][::-1], None, cv2.INTER_LINEAR,
                                     cv2.BORDER_REFLECT_101)
        mask = np.zeros_like(img2_cropped)
        cv2.fillConvexPoly(mask, np.int32(triangle2), (1, 1, 1), 16, 0)
        img2_cropped *= 1 - mask
        img2_cropped += img2_warped * mask
    return img2


@jit(target_backend='cuda')
def warp_portion(image, source_points, destination_points):
    print("Start Processing")
    result = np.zeros_like(image)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):

            offset = [0, 0]

            for source_point, destination_point in zip(source_points, destination_points):
                # point_position = (source_point[0] + destination_point[0], source_point[1] + destination_point[1])
                point_position = source_point
                shift_vector = destination_point
                point1 = (x, y)
                point2 = point_position
                vector = point1[0] - point2[0], point1[1] - point2[1]
                point_distance = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
                vector_length = math.sqrt(shift_vector[0] ** 2 + shift_vector[1] ** 2)
                helper = 1.0 / (3 * (point_distance / vector_length) ** 4 + 1)

                offset[0] -= helper * shift_vector[0]
                offset[1] -= helper * shift_vector[1]
            clamp1 = max(min(x + int(offset[0]), image.shape[1] - 1), 0)
            clamp2 = max(min(y + int(offset[1]), image.shape[0] - 1), 0)
            coords = (clamp1, clamp2)

            result[x, y] = image[coords[0], coords[1]]

    return result
