"""
Mask R-CNN
Configurations and data loading code for the synthetic Shapes dataset.
This is a duplicate of the code in the noteobook train_shapes.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.
Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

from https://github.com/matterport/Mask_RCNN/blob/master/samples/shapes/shapes.py

Heavily modified by marthinwurer
"""
from torch.utils.data import Dataset

"""
Mask R-CNN

The MIT License (MIT)

Copyright (c) 2017 Matterport, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import math
import random
import numpy as np
import cv2


PALLETE = [
#     [0, 0, 0],
#     [170, 0, 0],
#     [0, 170, 0],
#     [0, 0, 170],
#     [0, 170, 170],
#     [170, 0, 170],
#     [170, 170, 0],
#     [170, 170, 170],
#     [255, 255, 0],
#     [255, 0, 255],
#     [0, 255, 255],
    [0, 0, 255],
    [0, 255, 0],
    [255, 0, 0],
    [255, 255, 255],
]


class ShapesDataset:
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_shapes(self, count, height, width):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "square")
        self.add_class("shapes", 2, "circle")
        self.add_class("shapes", 3, "triangle")

        self.images = []

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in range(count):
            bg_color, shapes = self.random_image(height, width)
            self.add_image("shapes", image_id=i, path=None,
                           width=width, height=height,
                           bg_color=bg_color, shapes=shapes)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
        image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
        image = image * bg_color.astype(np.uint8)
        for shape, color, dims in info['shapes']:
            image = self.draw_shape(image, shape, dims, color)
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        shapes = info['shapes']
        count = len(shapes)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (shape, _, dims) in enumerate(info['shapes']):
            mask[:, :, i:i + 1] = self.draw_shape(mask[:, :, i:i + 1].copy(),
                                                  shape, dims, 1)
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(
                occlusion, np.logical_not(mask[:, :, i]))
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return mask, class_ids.astype(np.int32)

    def draw_shape(self, image, shape, dims, color):
        """Draws a shape from the given specs."""
        # Get the center x, y and the size s
        x, y, s = dims
        if shape == 'square':
            image = cv2.rectangle(image, (x - s, y - s),
                                  (x + s, y + s), color, -1)
        elif shape == "circle":
            image = cv2.circle(image, (x, y), s, color, -1)
        elif shape == "triangle":
            points = np.array([[(x, y - s),
                                (x - s / math.sin(math.radians(60)), y + s),
                                (x + s / math.sin(math.radians(60)), y + s),
                                ]], dtype=np.int32)
            image = cv2.fillPoly(image, points, color)
        return image

    def random_shape(self, height, width):
        """Generates specifications of a random shape that lies within
        the given height and width boundaries.
        Returns a tuple of three valus:
        * The shape name (square, circle, ...)
        * Shape color: a tuple of 3 values, RGB.
        * Shape dimensions: A tuple of values that define the shape size
                            and location. Differs per shape type.
        """
        # Shape
        shape = random.choice(["square", "circle", "triangle"])
        # Color
        # color = tuple([random.randint(0, 255) for _ in range(3)])
        color = tuple(random.choice(PALLETE))
        # Center x, y
        buffer = 20
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
        # Size
        s = random.randint(buffer, height // 4)
        return shape, color, (x, y, s)

    def random_image(self, height, width):
        """Creates random specifications of an image with multiple shapes.
        Returns the background color of the image and a list of shape
        specifications that can be used to draw the image.
        """
        # Pick random background color
        # bg_color = np.array([random.randint(0, 255) for _ in range(3)])
        bg_color = np.array([170, 170, 170])
        # Generate a few random shapes and record their
        # bounding boxes
        shapes = []
        boxes = []
        N = random.randint(1, 4)
        for _ in range(N):
            shape, color, dims = self.random_shape(height, width)
            shapes.append((shape, color, dims))
            x, y, s = dims
            boxes.append([y - s, x - s, y + s, x + s])
        # Apply non-max suppression wit 0.3 threshold to avoid
        # shapes covering each other
        keep_ixs = utils.non_max_suppression(
            np.array(boxes), np.arange(N), 0.3)
        shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
        return bg_color, shapes


def draw_shape(image, shape, dims, color):
    """Draws a shape from the given specs."""
    # Get the center x, y and the size s
    x, y, s = dims
    if shape == 'square':
        image = cv2.rectangle(image, (x - s, y - s),
                              (x + s, y + s), color, -1)
    elif shape == "circle":
        image = cv2.circle(image, (x, y), s, color, -1)
    elif shape == "triangle":
        points = np.array([[(x, y - s),
                            (x - s / math.sin(math.radians(60)), y + s),
                            (x + s / math.sin(math.radians(60)), y + s),
                            ]], dtype=np.int32)
        image = cv2.fillPoly(image, points, color)
    return image


def render_image(height, width, bg_color, shapes):
    """Generate an image from the specs of the given image ID.
    Typically this function loads the image from a file, but
    in this case it generates the image on the fly from the
    specs in image_info.
    """
    bg_color = np.array(bg_color).reshape([1, 1, 3])
    image = np.ones([height, width, 3], dtype=np.uint8)
    image = image * bg_color.astype(np.uint8)
    for shape, color, dims in shapes:
        image = draw_shape(image, shape, dims, color)
    return image


def random_shape(height, width):
    """Generates specifications of a random shape that lies within
    the given height and width boundaries.
    Returns a tuple of three valus:
    * The shape name (square, circle, ...)
    * Shape color: a tuple of 3 values, RGB.
    * Shape dimensions: A tuple of values that define the shape size
                        and location. Differs per shape type.
    """
    # Shape
    shape = random.choice(["square", "circle", "triangle"])
    # Color
#     color = tuple([random.randint(0, 255) for _ in range(3)])
    color = tuple(random.choice(PALLETE))
    # Center x, y
    buffer = 2
    y = random.randint(buffer, height - buffer - 1)
    x = random.randint(buffer, width - buffer - 1)
    # Size
    s = random.randint(buffer, height // 4)
    return shape, color, (x, y, s)


def random_image(height, width):
    """Creates random specifications of an image with multiple shapes.
    Returns the background color of the image and a list of shape
    specifications that can be used to draw the image.
    """
    # Pick random background color
#     bg_color = np.array([random.randint(0, 255) for _ in range(3)])
    bg_color = np.array([170, 170, 170])

    # Generate a few random shapes and record their
    # bounding boxes
    shapes = []
    boxes = []
    N = random.randint(1, 4)
    for _ in range(N):
        shape, color, dims = random_shape(height, width)
        shapes.append((shape, color, dims))
        x, y, s = dims
        boxes.append([y - s, x - s, y + s, x + s])

    return bg_color, shapes


def generate_image(height, width):
    bg_color, shapes = random_image(height, width)
    return render_image(height, width, bg_color, shapes)


def build_shape_dataset(height, width, count):
    images = [generate_image(height, width) for _ in range(count)]
    return images


class ShapeDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image
