from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np
import random


class DotActions(Enum):
    NONE = auto()
    DOWN = auto()
    RIGHT = auto()
    UP = auto()
    LEFT = auto()


class MovingDot(object):

    def __init__(self, width, height):
        self.center_x = width // 2
        self.center_y = height // 2
        self.width = width
        self.height = height
        self.x = random.randrange(0, width)
        self.y = random.randrange(0, height)

    def step(self, action: DotActions) -> (int, bool):
        """
        Do an action and return the reward
        Args:
            action: The DotAction to take on the world

        Returns:
            The reward for the action and whether the episode is done
        """
        old_x = self.x
        old_y = self.y

        if action is DotActions.DOWN:
            self.y += 1
        elif action is DotActions.UP:
            self.y -= 1
        elif action is DotActions.LEFT:
            self.x -= 1
        elif action is DotActions.RIGHT:
            self.x += 1

        self.x = max(0, min(self.width - 1, self.x))
        self.y = max(0, min(self.height - 1, self.y))

        before_distance = abs(self.center_x - old_x) + abs(self.center_y - old_y)
        after_distance = abs(self.center_x - self.x) + abs(self.center_y - self.y)

        return (before_distance - after_distance, self.x == self.center_x and self.y == self.center_y)

    def get_world(self):
        world = np.zeros((self.height, self.width))
        world[self.y][self.x] = 1.0
        return world


if __name__ == "__main__":
    dot = MovingDot(10, 10)
    for i in range(10):
        print(dot.step(DotActions.RIGHT))
        print(dot.get_world())

        world = dot.get_world()
        plt.imshow(world)
        plt.show()

