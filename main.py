from image_loader import load_random_image_from_project, display_image
from filters import box_filter, median_filter, gaussian_filter
import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_kernel_size():
    """
    Kullanıcıdan kernel boyutunu alır ve doğrular.
    """
    while True:
        try:
            kernel_size = int(input("Kernel boyutunu girin (tek sayı, örn: 3, 5, 7): "))
            if kernel_size % 2 == 0 or kernel_size <= 0:
                print("Lütfen pozitif bir **tek sayı** girin.")
                continue
            return kernel_size
        except ValueError:
            print("Geçerli bir sayı girin.")


def compare_filters(image, kernel_size):
    """
    Manuel filtreler ile OpenCV filtrelerini karşılaştırır.
    """
    # Manuel filtreler
    box_filtered_manual = box_filter(image, kernel_size=kernel_size)
    median_filtered_manual = median_filter(image, kernel_size=kernel_size)
    gaussian_filtered_manual = gaussian_filter(image, kernel_size=kernel_size, sigma=1)

    # OpenCV filtreleri
    box_filtered_opencv = cv2.blur(image, (kernel_size, kernel_size))
    median_filtered_opencv = cv2.medianBlur(image, kernel_size)
    gaussian_filtered_opencv = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=1.0)

    # Farkları hesapla
    box_diff = np.abs(box_filtered_manual - box_filtered_opencv)
    median_diff = np.abs(median_filtered_manual - median_filtered_opencv)
    gaussian_diff = np.abs(gaussian_filtered_manual - gaussian_filtered_opencv)

    # Görselleştir
    plt.figure(figsize=(20, 15))

    # Orijinal görüntü
    plt.subplot(4, 4, 1)
    plt.title("Orijinal")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    # Box filtresi karşılaştırma
    plt.subplot(4, 4, 2)
    plt.title("Box - Manuel")
    plt.imshow(box_filtered_manual, cmap="gray")
    plt.axis("off")
    plt.subplot(4, 4, 3)
    plt.title("Box - OpenCV")
    plt.imshow(box_filtered_opencv, cmap="gray")
    plt.axis("off")
    plt.subplot(4, 4, 4)
    plt.title(f"Box - Fark (Kernel: {kernel_size}x{kernel_size})")
    plt.imshow(box_diff, cmap="gray")
    plt.axis("off")

    # Median filtresi karşılaştırma
    plt.subplot(4, 4, 5)
    plt.title("Orijinal (Tekrar)")
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.subplot(4, 4, 6)
    plt.title("Median - Manuel")
    plt.imshow(median_filtered_manual, cmap="gray")
    plt.axis("off")
    plt.subplot(4, 4, 7)
    plt.title("Median - OpenCV")
    plt.imshow(median_filtered_opencv, cmap="gray")
    plt.axis("off")
    plt.subplot(4, 4, 8)
    plt.title(f"Median - Fark (Kernel: {kernel_size}x{kernel_size})")
    plt.imshow(median_diff, cmap="gray")
    plt.axis("off")

    # Gaussian filtresi karşılaştırma
    plt.subplot(4, 4, 9)
    plt.title("Orijinal (Tekrar)")
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.subplot(4, 4, 10)
    plt.title("Gaussian - Manuel")
    plt.imshow(gaussian_filtered_manual, cmap="gray")
    plt.axis("off")
    plt.subplot(4, 4, 11)
    plt.title("Gaussian - OpenCV")
    plt.imshow(gaussian_filtered_opencv, cmap="gray")
    plt.axis("off")
    plt.subplot(4, 4, 12)
    plt.title(f"Gaussian - Fark (Kernel: {kernel_size}x{kernel_size})")
    plt.imshow(gaussian_diff, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Piksel karşılaştırması
    print("Piksel Karşılaştırması (Manuel vs OpenCV):")
    for _ in range(10):
        x, y = np.random.randint(0, image.shape[0]), np.random.randint(0, image.shape[1])
        print(f"Pixel ({x}, {y}):")
        print(f"  Box - Manuel: {box_filtered_manual[x, y]} | OpenCV: {box_filtered_opencv[x, y]}")
        print(f"  Median - Manuel: {median_filtered_manual[x, y]} | OpenCV: {median_filtered_opencv[x, y]}")
        print(f"  Gaussian - Manuel: {gaussian_filtered_manual[x, y]} | OpenCV: {gaussian_filtered_opencv[x, y]}")
        print("-" * 40)


def main():
    try:
        # Proje klasöründeki rastgele bir görüntü yükle
        image, image_name = load_random_image_from_project()
        print(f"Yüklenen görüntü: {image_name}")

        # Kullanıcıdan kernel boyutunu al
        kernel_size = get_kernel_size()

        # Filtreleri çalıştır ve karşılaştır
        compare_filters(image, kernel_size)

    except Exception as e:
        print(f"Bir hata oluştu: {e}")


if __name__ == "__main__":
    main()