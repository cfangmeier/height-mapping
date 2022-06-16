from os.path import join
import cv2
import matplotlib.pyplot as plt
import numpy as np


class ImageSet:
    directory: str
    heights: [float]
    filenames: [str]

    raw_images: [np.ndarray]
    focus_value_maps: [np.ndarray]
    focus_avg_value_maps: [np.ndarray]

    all_in_focus_img: np.ndarray
    pixel_heights: np.ndarray

    SAVE_PLOTS = True

    def __init__(self, directory):
        from glob import glob
        from os.path import realpath
        self.directory = directory

        self.raw_images = []
        self.filenames = []
        self.heights = []
        for filename in glob(fr"{realpath(directory)}\img_*.png"):
            self.filenames.append(filename)
            self.raw_images.append(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))
            self.heights.append(float(filename[:-4].split('\\')[-1][4:]))

        self.calc_focus()
        self.all_in_focus()

    def show(self, which: str, colorscheme="gray"):
        img = getattr(self, which)
        fig = plt.figure(frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        plt.imshow(img, colorscheme)
        if self.SAVE_PLOTS:
            plt.savefig(join(self.directory, f'out_{which}.png'))
        else:
            plt.show()

    def show3d(self):
        fig = plt.figure(frameon=False)
        ax = fig.add_axes([0, 0, 1, 1], projection="3d")

        all_in_focus_img_color = cv2.cvtColor(self.all_in_focus_img, cv2.COLOR_GRAY2RGB) / 255.0
        m, n = self.pixel_heights.shape
        x, y = np.meshgrid(np.linspace(0, 1.4, n), np.linspace(0, 1.048, m))
        smoothed_pixel_heights = cv2.GaussianBlur(self.pixel_heights, (0, 0), 20)
        ax.plot_surface(x, y, smoothed_pixel_heights,
                        rcount=m//5,
                        ccount=n//5,
                        facecolors=all_in_focus_img_color,
                        )

        img_heights = np.asarray(self.heights)
        ax.set_zlim(max(img_heights), min(img_heights))
        ax.set_xlim(0, 1.4)
        ax.set_ylim(1.048, 0)
        if self.SAVE_PLOTS:
            plt.savefig(join(self.directory, f'out_3d.png'))
        else:
            plt.show()

    def save_mp4(self):
        from subprocess import run
        print("Saving sample images as mp4")
        magick = r"C:\Program Files\ImageMagick-7.1.0-Q16-HDRI\magick"
        cmd = f"{magick} convert -set delay 20 {' '.join(self.filenames)} {join(self.directory, 'out_scan.mp4')}"
        ret = run(cmd.split(' '))
        if ret.returncode:
            print(f"Command exited with error code {ret}")

    def calc_focus(self, gaussian_size: int = 11, laplacian_size: int = 11, avg_kernel_size: int = 9):
        self.focus_value_maps = []
        self.focus_avg_value_maps = []
        avg_kernel = np.ones((avg_kernel_size, avg_kernel_size))
        for raw_image in self.raw_images:
            gaussian_img = cv2.GaussianBlur(raw_image, (gaussian_size, gaussian_size), 0)
            laplacian_img = cv2.Laplacian(gaussian_img, cv2.CV_64F, ksize=laplacian_size)
            self.focus_value_maps.append(laplacian_img)
            self.focus_avg_value_maps.append(
                cv2.filter2D(laplacian_img * laplacian_img, -1, avg_kernel)
            )

    def all_in_focus(self):

        focus_measure = np.asarray(self.focus_avg_value_maps)
        argmax = np.argmax(focus_measure, axis=0)

        m, n = argmax.shape
        i, j = np.ogrid[:m, :n]
        raw_images = np.asarray(self.raw_images)
        self.all_in_focus_img = raw_images[argmax, i, j]

        img_heights = np.asarray(self.heights)
        self.pixel_heights = img_heights[argmax]


def main():
    for sample in [
        r"fid_focus_test_5\y_383_654",
        r"fid_focus_test_5\y_387_253",
        r"fid_focus_test_5\y_393_453",
    ]:
        image_set = ImageSet(f"data/{sample}")
        image_set.show("all_in_focus_img")
        image_set.show("pixel_heights", "viridis")
        image_set.show3d()
        image_set.save_mp4()


if __name__ == "__main__":
    main()