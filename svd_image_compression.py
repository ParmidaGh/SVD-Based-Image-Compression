import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

pic = 1

# 1- Save the image in a three-dimensional array
image = Image.open("D:\\درس\\ترم 3\\1-داده کاوی\\تمرین\\TA\\2\\Homework2\\Q6_IMGS\\"+str(pic)+".PPM")
img_array = np.array(image)
# print(img_array.shape)

# 2- Separate the RGB channels
r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]

# 3,4- SVD decomposition (low-rank approximation)
U_r, s_r, V_r = np.linalg.svd(r, full_matrices=False)
U_g, s_g, V_g = np.linalg.svd(g, full_matrices=False)
U_b, s_b, V_b = np.linalg.svd(b, full_matrices=False)


# 6- Run for each k
k = [50, 100, 150, 200]

for k in k:

    # 4- Approximation using the first k terms
    r_approx = U_r[:, :k] @ np.diag(s_r[:k]) @ V_r[:k, :]
    g_approx = U_g[:, :k] @ np.diag(s_g[:k]) @ V_g[:k, :]
    b_approx = U_b[:, :k] @ np.diag(s_b[:k]) @ V_b[:k, :]

    # 5- Merge the three resulting matrices
    img_approx = np.zeros(img_array.shape)
    img_approx[:, :, 0], img_approx[:, :, 1], img_approx[:, :, 2] = r_approx, g_approx, b_approx
    img_approx = img_approx.astype(int)

    plt.figure(figsize=(10, 5))

    # Original image
    plt.subplot(1, 2, 1)
    plt.title('Original image '+str(pic))
    plt.imshow(image)

    # Reconstructed image
    plt.subplot(1, 2, 2)
    plt.title(f'Compressed image (k={k})')
    plt.imshow(img_approx)

    plt.tight_layout()
    plt.show()