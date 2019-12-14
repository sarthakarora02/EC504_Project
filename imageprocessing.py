import cv2
import numpy as np
import matplotlib.pyplot as plt


def imageprocessing(mask, og_image):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask.astype('uint8'), connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    min_size = 3
    mask = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            mask[output == i + 1] = 255
    plt.figure()
    plt.imshow(mask, cmap='gray')

    kernel = np.ones((8,8), np.uint8)
    mask = cv2.dilate(mask.astype('uint8'), kernel)
    kernel = np.ones((6,6), np.uint8)
    mask = cv2.erode(mask, kernel)
    plt.figure()
    plt.imshow(mask, cmap='gray')

    mask = cv2.resize(mask, (300,300), interpolation = cv2.INTER_AREA)
    plt.figure()
    cv2.imshow('mask', mask)

    bk = cv2.bitwise_and(og_image, mask)
    bk = 255+bk
    plt.figure()
    cv2.imshow('segment', bk)

    k = cv2.waitKey(0) & 0xFF
    if k == ord('q'):
        cv2.destroyAllWindows()

    cv2.imwrite('mask.jpg', mask)
    cv2.imwrite('segmented_image.jpg', bk)
