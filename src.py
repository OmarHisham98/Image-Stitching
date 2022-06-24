import numpy as np
from numpy.core.shape_base import atleast_1d
import cv2
import matplotlib.pyplot as plt
from skimage import io
import math as M

# img1_BGR = cv2.imread('image1.jpg')
# img1_rgb = cv2.cvtColor(img1_BGR, cv2.COLOR_BGR2RGB)

# Create point matrix get coordinates of mouse click on image


point_matrix = np.zeros((4, 2), np.int)

counter = 0


def mousePoints(event, x, y, flags, params):
    global counter
    # Left button mouse click event opencv
    if event == cv2.EVENT_LBUTTONDOWN and counter < 4:
        point_matrix[counter] = x, y
        counter = counter + 1
        print("counter=", counter)


def get_points(image_path):
    # Read image
    img = cv2.imread(image_path)

    while counter < 5:
        for x in range(0, 4):
            cv2.circle(img, (point_matrix[x][0], point_matrix[x][1]), 3, (0, 255, 0), cv2.FILLED)

        if counter == 4:
            # starting_x = point_matrix[0][0]
            # starting_y = point_matrix[0][1]

            # ending_x = point_matrix[1][0]
            # ending_y = point_matrix[1][1]
            cv2.waitKey(10)
            break
            # Draw rectangle for area of interest
            # cv2.rectangle(img, (starting_x, starting_y), (ending_x, ending_y), (0, 255, 0), 3)

            # Cropping image
            # img_cropped = img[starting_y:ending_y, starting_x:ending_x]
            # cv2.imshow("ROI", img_cropped)

        # Showing original image
        cv2.imshow("Original Image ", img)
        # Mouse click event on original image
        cv2.setMouseCallback("Original Image ", mousePoints)
        # Printing updated point matrix

        # Refreshing window all time
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    print(point_matrix)


def create_a_matrix(point_matrix_11, point_matrix_22, crr_num):  # correspondence number
    a = np.zeros((2, 9), np.int)
    # a = [[ - point_matrix_11(crr_num,0) , - point_matrix_11(crr_num,1) , -1 , 0 , 0 , 0 , point_matrix_11(crr_num,0) * point_matrix_22(crr_num,0) , point_matrix_11(crr_num,1) * point_matrix_22(crr_num,0) , point_matrix_22(crr_num,0)],
    #      [ 0 , 0 , 0 , - point_matrix_11(crr_num,0) , - point_matrix_11(crr_num,1) , -1 , point_matrix_11(crr_num,0) * point_matrix_22(crr_num,1) , point_matrix_11(crr_num,1) * point_matrix_22(crr_num,1) , point_matrix_22(crr_num,1)]]

    a[0][0] = - point_matrix_11[crr_num][0]
    a[0][1] = - point_matrix_11[crr_num][1]
    a[0][2] = -1
    a[0][6] = point_matrix_11[crr_num][0] * point_matrix_22[crr_num][0]
    a[0][7] = point_matrix_11[crr_num][1] * point_matrix_22[crr_num][0]
    a[0][8] = point_matrix_22[crr_num][0]

    a[1][3] = - point_matrix_11[crr_num][0]
    a[1][4] = - point_matrix_11[crr_num][1]
    a[1][5] = -1
    a[1][6] = point_matrix_11[crr_num][0] * point_matrix_22[crr_num][1]
    a[1][7] = point_matrix_11[crr_num][1] * point_matrix_22[crr_num][1]
    a[1][8] = point_matrix_22[crr_num][1]
    print("a = " , a)
    return a


def compute_homography():
    a1 = create_a_matrix(point_matrix_1, point_matrix_2, 0)
    a2 = create_a_matrix(point_matrix_1, point_matrix_2, 1)
    a3 = create_a_matrix(point_matrix_1, point_matrix_2, 2)
    a4 = create_a_matrix(point_matrix_1, point_matrix_2, 3)

    a = np.concatenate((a1, a2, a3, a4), axis=0)
    print("a=", a)
    u, s, vh = np.linalg.svd(a)
    print("u=", u)
    print("s=", s)
    print("vh=", vh)
    print("vh shape=", np.shape(vh))
    h = vh[-1].reshape((3,3)) # take last column + reshape it to 3x3
    h = h / h[2][2]
    print("h shape=", np.shape(h))
    print("h=", h)

    return h


def invWarping(warpedImage, originalImage, homography):
    h_inverse = np.linalg.inv(homography)
    for i in range(warpedImage.shape[0]):
        for j in range(warpedImage.shape[1]):
            point = np.array([i, j, 1])
            # new_points = np.reshape(point, (3, 1))
            # print(new_points.shape)
            # print(new_points)

            newPoints = h_inverse.dot(point)
            x_original = float(newPoints[0] / newPoints[2])
            y_original = float(newPoints[1] / newPoints[2])

            if np.all(warpedImage[i][j] == 0) :
                x2 = int(M.ceil(x_original))
                y2 = int(M.ceil(y_original))
                x1 = int(M.floor(x_original))
                y1 = int(M.floor(y_original))
                p1 = [x1, y1]  # smaller x and y
                p2 = [x2, y1]  # bigger x and smaller y
                p3 = [x1, y2]  # smaller x and bigger y
                p4 = [x2, y2]  # bigger x and y
                weight1 = (1 - (x_original - p1[0])) * (1 - (y_original - p1[1]))
                weight2 = (1 - (p2[0] - x_original)) * (1 - (y_original - p2[1]))
                weight3 = (1 - (x_original - p3[0])) * (1 - (p3[1] - y_original))
                weight4 = (1 - (p4[1] - x_original)) * (1 - (p4[1] - y_original))

                if x2 < originalImage.shape[0] and y2 < originalImage.shape[1] and x1>=0 and x2>=0 :

                    new_rgb = originalImage[p1[0]][p1[1]] * weight1 + originalImage[p2[0]][p2[1]] * weight2 + \
                          originalImage[p3[0]][p3[1]] * weight3 + originalImage[p4[0]][p4[1]] * weight4

                    warpedImage[i][j] = new_rgb

    print(warpedImage)
    cv2.imshow("inverse image", warpedImage.astype(np.uint8))
    cv2.waitKey(0)


def warpingImage(sourceImg, homography, destImg):
    warpedIMage = np.zeros((destImg.shape[0]+500, destImg.shape[1]+500, 3), dtype=int)

    for i in range(sourceImg.shape[0]):
        for j in range(sourceImg.shape[1]):
            point = np.array([i, j, 1])
            # new_points = np.reshape(point, (3, 1))
            # print(new_points.shape)
            # print(new_points)

            newPoints = np.dot(homography, point)
            x_dash = int(newPoints[0] / newPoints[2])
            y_dash = int(newPoints[1] / newPoints[2])
            if destImg.shape[0]+500 > x_dash and x_dash >= 0 and y_dash < destImg.shape[1]+500 and y_dash >= 0:
                warpedIMage[x_dash][y_dash] = sourceImg[i][j]

    print(warpedIMage)


    return warpedIMage


def stitchImages(origImg, transformedImg, shiftX, shiftY, nChannels = 3):
  newImg = np.zeros(((origImg.shape[0] + transformedImg.shape[0]), (origImg.shape[1] + transformedImg.shape[1]), nChannels))
  newImg[0:transformedImg.shape[0],0:transformedImg.shape[1]] = transformedImg
  for row in range(origImg.shape[0]):
    for col in range(origImg.shape[1]):
      newX, newY = col + shiftX , row + shiftY
      newImg[newY][newX] = origImg[row][col]
  # delete black rows & columns
  idx = np.argwhere(np.all(newImg[..., :] == 0, axis=0))
  newImg = np.delete(newImg, idx, axis=1)
  idx = np.argwhere(np.all(newImg[..., :] == 0, axis=1))
  newImg = np.delete(newImg, idx, axis=0)
  # rotate image columns if shift was -ve
  if shiftX < 0:
    newImg = np.roll(newImg, -1* shiftX, axis=1)
  if shiftY < 0:
    newImg = np.roll(newImg, -1* shiftY, axis=0)
  return newImg.astype(np.uint8)




if __name__ == "__main__":
    get_points("image1.jpg")
    point_matrix_1 = np.copy(point_matrix)
    counter = 0
    get_points("image2.jpg")
    point_matrix_2 = np.copy(point_matrix)
    H = compute_homography()
    h = [[1, 0, 12],
         [0, 1, 12],
         [0, 0, 1]]


    img1 = cv2.imread("image1.jpg")
    img2 = cv2.imread("image2.jpg")
    print("img1 shape=", np.shape(img1))
    wimg = warpingImage(img1, H, img2)
    # translation = [[1, 0, 100],
    #                [0, 1, -100],
    #                [0, 0, 1]]

    
    cv2.imshow("Warped image", wimg.astype(np.uint8))
    cv2.waitKey(0)
    invWarping(warpedImage=wimg, originalImage=img1, homography=H)
