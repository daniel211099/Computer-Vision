import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread


def affine_transformation(m_a, a0, img0):
    img_transformed = np.empty((1000, 1000, 3), dtype=np.uint8)
    for i, row in enumerate(img0):
        for j, col in enumerate(row):
            pixel_data = img0[i, j, :]
            input_coords = np.array([i, j, 1])
            i_out, j_out, _ = (m_a @ input_coords)
            i_out = i_out + a0[0]
            j_out = j_out + a0[1]
            i_out = int(i_out)
            j_out = int(j_out)
            img_transformed[i_out, j_out, :] = pixel_data

    return img_transformed


def plot_image(titel, img):
    plt.figure(figsize=(5, 5))
    plt.title(titel)
    plt.imshow(img)
    plt.show()


image = imread('C:\Studium_MSI\Computer Vision\Aufgabe1/gletscher.jpg')

print(image.shape)

plot_image("Orginal Bild",image)

a1 = np.array([[0], [0], [0]])

# (1) Rotation des Bildes um 30 Grad
phi = (30 / 360) * 2 * np.pi
sc = 0.7
T_1 = np.array([[np.cos(phi), np.sin(phi), 0], [-np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
a2 = np.array([[0], [250], [0]])
img_1 = affine_transformation(T_1, a2, image)
plot_image("(1) Drehen um 30 Grad", img_1)

# (2) Verkleinern des Bildes um Faktor 0,7
T_2 = np.array([[0.7, 0, 0], [0, 0.7, 0], [0, 0, 1]])  # Identity matrix
img_2 = affine_transformation(T_2, a1, image)
plot_image("(2) Verkleinern um Faktor 0,7", img_2)

# (3) Verkleinern in x (0,7) vergrößern in y (1,2)
T_3 = np.array([[0.8, 0, 0], [0, 1.2, 0], [0, 0, 1]])
img_3 = affine_transformation(T_3, a1, image)
plot_image("(3) Verkleinern in x (0,7) vergrößern in y (1,2)", img_3)

# (4) Dehnen um 1.5 und senkrecht stauchen um 0.5Spiegeln entlang der Diagonalen
T_4 = np.array([[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1]])
img_4 = affine_transformation(T_4, a1, image)
T_41 = np.array([[1, 0, 0], [0, 0.5, 0], [0, 0, 1]])
img_41 = affine_transformation(T_41, a1, img_4)
plot_image("(4) Dehnen um 1.5 und senkrecht stauchen um 0.5", img_41)

print("Press Enter to exit")
