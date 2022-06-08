from PIL import Image
import os
import torchvision.transforms as transforms

class TransformImage:

    """
    Performs a resize to get one side to 640px and scale the other one appropriate to keep the dimensions
    In the next step it will apply padding to the remaining side
    """

    def __init__(self):
        self._target_height = 640
        self._target_width = 640

    def difference(self, img_size):
        w, h = img_size
        h_diff = (self._target_height - h) // 2
        w_diff = (self._target_width - w) // 2
        return w_diff, h_diff

    def resize_and_pad(self, img):
        w, h = img.size
        w_diff, h_diff = self.difference(img.size)
        if h_diff > 0 or w_diff > 0:
            width_required = self._target_width
            height_required = self._target_height
            if h_diff < w_diff:
                scale = self._target_height / h
                width_required = round(w * scale)
            elif w_diff < h_diff:
                scale = self._target_width / w
                height_required = round(h * scale)
            resize = transforms.Resize((height_required, width_required))
        else:
            resize = transforms.Resize((self._target_width, self._target_height))
        img = resize(img)
        w_diff, h_diff = self.difference(img.size)
        transform = transforms.Pad((w_diff, h_diff))
        img = transform(img)
        img.show()


if __name__ == "__main__":
    resiz = TransformImage()
    path = 'Dataset/TFEID High/dfh_fear_x/'
    img_name = "S010_004_00000019.png"
    list = os.listdir(path)
    print(list)
    for name in list:
        img = Image.open(path + name)
        resiz.resize_and_pad(img)