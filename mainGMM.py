import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.mixture import GaussianMixture as gmm
import math
from queue import Queue
import math
import datetime

from edmondskarp import EdmondsKarp
from imageprocessing import imageprocessing
from makegraphGMM import makegraphGMM

def createLineIterator(P1, P2, img):
    """
    Source: https://stackoverflow.com/questions/32328179/opencv-3-0-lineiterator
    """
    #define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]

    #difference and absolute difference between points
    #used to calculate slope and relative location between points
    dX = P2[0] - P1[0]
    dY = P2[1] - P1[1]
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    #predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
    itbuffer.fill(np.nan)

    #Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1[1] > P2[1]
    negX = P1[0] > P2[0]
    if P1[0] == P2[0]: #vertical line segment
        itbuffer[:,0] = P1[0]
        if negY:
            itbuffer[:,1] = np.arange(P1[1] - 1,P1[1] - dYa - 1,-1)
        else:
            itbuffer[:,1] = np.arange(P1[1]+1,P1[1]+dYa+1)
    elif P1[1] == P2[1]: #horizontal line segment
        itbuffer[:,1] = P1[1]
        if negX:
            itbuffer[:,0] = np.arange(P1[0]-1,P1[0]-dXa-1,-1)
        else:
            itbuffer[:,0] = np.arange(P1[0]+1,P1[0]+dXa+1)
    else: #diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32)/dY.astype(np.float32)
            if negY:
                itbuffer[:,1] = np.arange(P1[1]-1,P1[1]-dYa-1,-1)
            else:
                itbuffer[:,1] = np.arange(P1[1]+1,P1[1]+dYa+1)
            itbuffer[:,0] = (slope*(itbuffer[:,1]-P1[1])).astype(np.int) + P1[0]
        else:
            slope = dY.astype(np.float32)/dX.astype(np.float32)
            if negX:
                itbuffer[:,0] = np.arange(P1[0]-1,P1[0]-dXa-1,-1)
            else:
                itbuffer[:,0] = np.arange(P1[0]+1,P1[0]+dXa+1)
            itbuffer[:,1] = (slope*(itbuffer[:,0]-P1[0])).astype(np.int) + P1[1]

    #Remove points outside of image
    colX = itbuffer[:,0]
    colY = itbuffer[:,1]
    itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

    #Get intensities from img ndarray
    itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]

    return itbuffer


def on_mouse(event, x, y, flag, params):
    global cords, draw, count
    if event == cv2.EVENT_LBUTTONDOWN:
        cords.append((x,y))
        draw = True
    elif event == cv2.EVENT_LBUTTONUP:
        cords.append((x,y))
        cv2.line(image, cords[count], cords[count+1], color=(255,255,255), thickness=1)
        cv2.imshow('image', image)
        count+=2
        draw = False

def get_pixels(cords, img):
    pix = createLineIterator(np.array(cords[0]), np.array(cords[1]), og_image)
    return pix[:,2].astype('int')


def displayCut(image, cuts):
    def colorPixel(i, j):
#         print(image.shape)
        image[i][j] = CUTCOLOR

    r, c = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for c in cuts:
        if c[0] != SOURCE and c[0] != SINK and c[1] != SOURCE and c[1] != SINK:
            # print("cut")
            colorPixel(c[0] // r, c[0] % r)
            colorPixel(c[1] // r, c[1] % r)
    return image


def imageSegmentation(image, fore_pixels, back_pixels):
    # print("Enter the Seg: ", image.shape)
    image = cv2.resize(image, (30,30))
    # print(image)
    graph, seededImage = makegraphGMM(image, fore_pixels, back_pixels)

    global SOURCE, SINK
    SOURCE += len(graph)
    SINK   += len(graph)

    start = datetime.datetime.now()
    cuts = EdmondsKarp(graph, SOURCE, SINK)
    stop = datetime.datetime.now()
    print("Time Taken: ", stop-start)
    # print("cuts:")
    # print(cuts)
    image = displayCut(image, cuts)
    image = cv2.resize(image, (0, 0), fx=Scale, fy=Scale)
#     show_image(image)
    savename = "cut.jpg"
    cv2.imwrite(savename, image)
    print("Saved image as", savename)
    return image, cuts


if __name__ == "__main__":
    cords = []
    draw = False
    count = 0
    # SIGMA = 30
    OBJCOLOR, BKGCOLOR = (0, 0, 255), (0, 255, 0)

    CUTCOLOR = (0, 0, 255)

    SOURCE, SINK = -2, -1
    Scale = 10

    parser = argparse.ArgumentParser()
    parser.add_argument("imagefile")
    # print(parser.parse_args().imagefile)

    og_image = cv2.imread(parser.parse_args().imagefile, 0)
    # og_image = cv2.imread('./birdy.jpeg', 0)
    # og_image = cv2.imread('./eagle.jpeg', 0)
    # og_image = cv2.imread('./plane.jpeg', 0)
    # og_image = cv2.imread('./diff.jpeg', 0)
    og_image = cv2.resize(og_image, (300,300), interpolation = cv2.INTER_AREA)
    # og_image = cv2.resize(og_image, (300,300))
    image = og_image.copy()

    print("Mark foreground \nPress b to mark background")

    cv2.namedWindow("image")
    cv2.setMouseCallback('image', on_mouse)
    while True:
        cv2.imshow('image', image)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('b'):
            fore_count = count
            print("\n\nMark background \nPress q to quit marking")
        if k == ord('q'):
            cv2.imwrite('seededimg.png',image)
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()

    fore_cords = cords[:fore_count]
    back_cords = cords[fore_count:]

    fore_pixels = []
    back_pixels = []
    i = 0
    while i < len(fore_cords):
        pix = createLineIterator(np.array(fore_cords[i]), np.array(fore_cords[i+1]), og_image)
        fore_pixels+=list(pix)
        i+=2
    j = 0
    while j < len(back_cords):
        pix = createLineIterator(np.array(back_cords[j]), np.array(back_cords[j+1]), og_image)
        back_pixels+=list(pix)
        j+=2

    fore_pixels = np.array(fore_pixels)
    fore_pixels = fore_pixels[:,:2].astype('int32')
    back_pixels = np.array(back_pixels)
    back_pixels = back_pixels[:,:2].astype('int32')

    image, cuts = imageSegmentation(og_image, fore_pixels, back_pixels)

    cut_img = cv2.imread('./cut.jpg')
    # plt.figure()
    cv2.imshow('cut_image', cut_img)
    # plt.show()

    mask = np.zeros((30,30)).astype('int32')
    pts = []
    for i, val in enumerate(cuts):
        if val[0] != SOURCE and val[0] != SINK and val[1] != SOURCE and val[1] != SINK:
            mask[val[1]//30][val[0]%30] = 255
            pts.append([val[1]%30,val[0]//30])

    pts = np.array(pts, dtype='int32')

    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.show()

    test = np.zeros((30,30)).astype('int32')
    fill = cv2.fillConvexPoly(test, pts, 255, lineType=8)

    plt.figure()
    plt.imshow(fill, cmap='gray')
    plt.show()

    fill = cv2.resize(fill.astype('uint8'), (300,300))

    plt.figure()
    plt.imshow(fill*og_image.astype('int32'), cmap='gray')
    plt.show()
