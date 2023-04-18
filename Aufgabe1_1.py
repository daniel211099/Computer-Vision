import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from skimage.io import imread


def bilinear_interpolation(image, x, y):
    x1 = int(x)
    y1 = int(y)
    x2 = x1 + 1
    y2 = y1 + 1

    if x2 >= image.shape[0]:
        return 0
    if y2 >= image.shape[1]:
        return 0

    P1 = image[x1, y2]
    P2 = image[x2, y2]
    P3 = image[x1, y1]
    P4 = image[x2, y1]

    A1 = (abs(x - x1) * abs(y2 - y)) / (abs(y2 - y1) * abs(x2 - x1))
    A2 = (abs(x2 - x) * abs(y2 - y)) / (abs(y2 - y1) * abs(x2 - x1))
    A3 = (abs(x - x1) * abs(y - y1)) / (abs(y2 - y1) * abs(x2 - x1))
    A4 = (abs(x2 - x) * abs(y - y1)) / (abs(y2 - y1) * abs(x2 - x1))

    return A4 * P1 + A3 * P2 + A2 * P3 + A1 * P4


def naechsterNachbar(iO, jO, img0):
    iO = int(iO)
    jO = int(jO)
    if iO > img0.shape[0] - 1:
        pixel_data = [0, 0, 0]
    elif jO > img0.shape[1] - 1:
        pixel_data = [0, 0, 0]
    else:
        pixel_data = img0[iO, jO, :]
    return pixel_data


def getNewImageSize(m_a, img0, a0):
    x_size = int(m_a[0, 0] * img0.shape[0] + m_a[0, 1] * img0.shape[1] + m_a[0, 2] + a0[0] + 2) +100
    y_size = int(m_a[1, 0] * img0.shape[0] + m_a[1, 1] * img0.shape[1] + m_a[1, 2] + a0[1] + 2) +100
    return x_size, y_size


def affine_transformation(m_a, a0, img0, interpolation):
    x_size, y_size = getNewImageSize(m_a, img0, a0)
    img_transformed = np.empty((x_size, y_size, 3), dtype=np.uint8)
    m_a = inv(m_a)
    for i, row in enumerate(img_transformed):
        for j, col in enumerate(row):
            input_coords = np.array([i, j, 1])
            i_out, j_out, _ = (m_a @ input_coords)
            if input_coords[0] + a0[0] > img_transformed.shape[0] - 1 or input_coords[1] + a0[1] > img_transformed.shape[1] - 1 :
                continue
            if i_out < 0 or j_out < 0:
                continue
            if interpolation:
                # Bilinaear interpolation
                img_transformed[i + a0[0], j + a0[1], :] = bilinear_interpolation(img0, i_out, j_out)
            else:
                # Nächster Nachbar
                img_transformed[i + a0[0], j + a0[1], :] = naechsterNachbar(i_out, j_out, img0)

    return img_transformed


def plot_image(titel, img):
    plt.figure(figsize=(5, 5))
    plt.title(titel)
    plt.imshow(img)


def bildAmbassador():
    # (5) Entzerren des Bildes Ambassadors
    image = imread('C:\Studium_MSI\Computer Vision\Aufgabe1/ambassadors.jpg')
    T_5 = np.array([[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1]])
    img_5 = affine_transformation(T_5, [0, 0, 0], image, False)
    plot_image("Ambassadors mit nächster Nachbar", img_5)
    img_5 = affine_transformation(T_5, [0, 0, 0], image, True)
    plot_image("Ambassadors mit bilinearer Interpolation", img_5)


def taskd1(image):
    print("Aufgabe 1d (1) Rotation des Bildes um 30 Grad")
    phi = (30 / 360) * 2 * np.pi
    a = np.array([[0], [0], [0]])
    T = np.array([[np.cos(phi), np.sin(phi), 0], [-np.sin(phi), np.cos(phi), 250], [0, 0, 1]])
    img = affine_transformation(T, a, image, False)
    plot_image("(1) Drehen um 30 Grad", img)
    img = affine_transformation(T, a, image, True)
    plot_image("(1) Drehen um 30 Grad Interpolation", img)


def taskd2(image):
    print("Aufgabe 1d (2) Verkleinern des Bildes um Faktor 0,7")
    T = np.array([[0.7, 0, 0], [0, 0.7, 0], [0, 0, 1]])
    img = affine_transformation(T, [0, 0, 0], image, False)
    plot_image("(2) Verkleinern um Faktor 0,7", img)
    img = affine_transformation(T, [0, 0, 0], image, True)
    plot_image("(2) Verkleinern um Faktor 0,7 Interpolation", img)


def taskd3(image):
    print("Aufgabe 1d (3) Verkleinern in x (0,7) vergrößern in y (1,2)")
    T = np.array([[0.8, 0, 0], [0, 1.2, 0], [0, 0, 1]])
    img = affine_transformation(T, [0, 0, 0], image, False)
    plot_image("(3) Verkleinern in x (0,7) vergrößern in y (1,2)", img)
    img = affine_transformation(T, [0, 0, 0], image, True)
    plot_image("(3) Verkleinern in x (0,7) vergrößern in y (1,2) Interpolation", img)


def taskd4(image):
    print("Aufgabe 1d  (4) Dehnen um 1.5 und senkrecht stauchen um 0.5Spiegeln entlang der Diagonalen")
    T_1 = np.array([[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1]])
    img1 = affine_transformation(T_1, [0, 0, 0], image, True)
    T_2 = np.array([[1, 0, 0], [0, 0.5, 0], [0, 0, 1]])
    img1 = affine_transformation(T_2, [0, 0, 0], img1, True)
    plot_image("(4) Dehnen um 1.5 und senkrecht stauchen um 0.5", img1)


gletscherImg = imread('C:\Studium_MSI\Computer Vision\Aufgabe1/gletscher.jpg')
plot_image("Orginal bild", gletscherImg)

# Aufgabe 1 b)
T_translate = np.array([[1, 0, 20], [0, 1, 20], [0, 0, 1]])
img1 = affine_transformation(T_translate, [0, 0, 0], gletscherImg, False)
plot_image("Aufgabe 1 b)", img1)
plt.show()


# Aufgabe 1 c)
gletscher30Img = imread('C:\Studium_MSI\Computer Vision\Aufgabe1/gletscher30p.jpg')
plot_image("Orginal Bild",gletscher30Img)
T_1 = np.array([[10.5, 0, 0], [0, 10.5, 0], [0, 0, 1]])
img1 = affine_transformation(T_1, [0, 0, 0], gletscher30Img, False)
plot_image("Aufgabe c) Nächster Nachbar", img1)
img1 = affine_transformation(T_1, [0, 0, 0], gletscher30Img, True)
plot_image("Aufgabe c) Bilineare Interpolation", img1)
plt.show()
#
# Affine Transformationen Aufgabe d)
taskd1(gletscherImg)
plt.show()
taskd2(gletscherImg)
plt.show()
taskd3(gletscherImg)
plt.show()
taskd4(gletscherImg)
plt.show()
bildAmbassador()
plt.show()
