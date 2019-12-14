import numpy as np
import cv2
from sklearn.mixture import GaussianMixture as gmm

from boundaryPenalty import boundaryPenalty

def makegraphGMM(image, fore_pixels, back_pixels):
    V = image.size + 2
    # graph = np.zeros((V, V))
    graph = np.zeros((V, V), dtype='uint8')
#     print(graph.shape)
    K = addNedges(graph, image)
    seeds, seededImage = plantSeed(image, fore_pixels, back_pixels)
    addTedges(image,graph, seeds, K)
    return graph, seededImage


def addNedges(graph, image):
    K = -float("inf")
    # edges = cv2.Canny(image,100,200)
    # print(image)
    # print(edges.shape)
    countN = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            x = i * image.shape[1] + j
            if i + 1 < image.shape[0]: # pixel below
                y = (i + 1) * image.shape[1] + j
                # print("1: ",x,y)
                bp = boundaryPenalty(image[i][j], image[i + 1][j])
                # bp = boundaryPenalty(image[i][j], image[i + 1][j], (i,j),(i+1,j))
                graph[x][y] = graph[y][x] = bp
                # print(bp)
                K = max(K, bp)
                countN = countN + 1
            if j + 1 < image.shape[1]: # pixel to the right
                y = i * image.shape[1] + j + 1
                # print("2: ",x,y)
                bp = boundaryPenalty(image[i][j], image[i][j + 1])
                # bp = boundaryPenalty(image[i][j], image[i][j + 1], (i,j),(i,j+1))
                graph[x][y] = graph[y][x] = bp
                K = max(K, bp)
                countN = countN + 1
            # if j + 1 < c and i+1<r: # pixel to the bottom right
            #     y = (i+1) * c + (j + 1)
            #     bp = boundaryPenalty(image[i][j], image[i+1][j + 1])
            #     graph[x][y] = graph[y][x] = bp
            #     K = max(K, bp)
            #     print("3: ",x,y)

    print("N-edges: ",countN)
    return K


def plantSeed(image, fore_pixels, back_pixels):
    Scale = 10
    seeds = np.zeros((image.shape[0],image.shape[1]), dtype='uint8')
    for n1, val1 in enumerate(fore_pixels):
        seeds[val1[1]//Scale][val1[0]//Scale] = 1
    for n2, val2 in enumerate(back_pixels):
        seeds[val2[1]//Scale][val2[0]//Scale] = 2
#     seeds = cv2.resize(seeds, (30,30))
    seededImage = cv2.imread('./seededimg.png', 0)
    return seeds, seededImage

def addTedges(image,graph, seeds, K):
    OBJCODE, BKGCODE = 1, 2
    SOURCE, SINK = -2, -1
    X = []
    Y = []
    for i in range(seeds.shape[0]):
        for j in range(seeds.shape[1]):
            if seeds[i][j] !=0:
                X.append(image[i][j])
                if seeds[i][j] == 2:
                    Y.append(0)
                if seeds[i][j] == 1:
                    Y.append(1)
    gauss = gmm(n_components=2, covariance_type='tied', init_params='kmeans', verbose=1, verbose_interval=10, random_state = 24)
    X = np.array(X)
    Y = np.array(Y)
    X = X.reshape(len(X),1)
    Y = Y.reshape(len(Y),1)
    # print(X.shape, Y.shape)
    gauss = gauss.fit(X,Y)
    prob_labels = gauss.predict_proba(image.reshape((image.size,1)))
    # print("GMM: ", prob_labels)

    countT = 0
    for i in range(seeds.shape[0]):
        for j in range(seeds.shape[1]):
            x = i * seeds.shape[1] + j
            if seeds[i][j] == OBJCODE:
                graph[SOURCE][x] = K
                # graph[SOURCE][x] = np.log(K*prob_labels[x][1]/prob_labels[x][0])
                # print(np.log(K*prob_labels[x][1]/prob_labels[x][0]), K)
                countT = countT + 1
            elif seeds[i][j] == BKGCODE:
                graph[x][SINK] = K
                countT = countT + 1
                # graph[x][SINK] = np.log(K*prob_labels[x][1]/prob_labels[x][0])
                # print(np.log(K*prob_labels[x][1]/prob_labels[x][0]), K)
            else:
                # graph[x][SINK] = K
                # graph[SOURCE][x] = K
                if np.log(K*prob_labels[x][1]/prob_labels[x][0])> 0 and np.log(K*prob_labels[x][1]/prob_labels[x][0]) != np.Inf:
                    # graph[x][SINK] = -np.log(K*prob_labels[x][0]/prob_labels[x][1])
                    graph[SOURCE][x] = np.log(K*prob_labels[x][1]/prob_labels[x][0])
                    # print(np.log(K*prob_labels[x][1]/prob_labels[x][0]))
                    countT = countT + 1
                elif np.log(K*prob_labels[x][1]/prob_labels[x][0]) < 0 and np.log(K*prob_labels[x][1]/prob_labels[x][0]) != np.Inf:
                    # graph[SOURCE][x] = -np.log(K*prob_labels[x][1]/prob_labels[x][0])
                    graph[x][SINK] = np.log(K*prob_labels[x][0]/prob_labels[x][1])
                    # print(np.log(K*prob_labels[x][0]/prob_labels[x][1]))
                    countT = countT + 1

    print("T-edges: ", countT)
