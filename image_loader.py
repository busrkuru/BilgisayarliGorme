import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def load_random_image_from_project():
    """
    Proje klasöründeki images klasöründen rastgele bir görüntü yükler.
    """
    # Proje içindeki images klasörünü belirle
    image_folder = "images"

    # Görüntü dosyalarını listele
    images = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
    if not images:
        raise FileNotFoundError("Hata: 'images' klasöründe görüntü bulunamadı!")

    # Rastgele bir görüntü seç
    selected_image = np.random.choice(images)
    image_path = os.path.join(image_folder, selected_image)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise IOError(f"Hata: '{selected_image}' dosyası yüklenemedi!")
    return image, selected_image

def display_image(image, title="Görüntü"):
    """
    Görüntüyü matplotlib ile gösterir.
    """
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()