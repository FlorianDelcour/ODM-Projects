"""
    University of Liege
    INFO8003-1 - Optimal decision making for complex problems
    Assignment 2 - Reinforcement Learning in a Continuous Domain
    Authors : 
        DELCOUR Florian
        MAKEDONSKY Aliocha
"""

import random
import imageio
from section1 import *
from car_on_the_hill_images import save_caronthehill_image
import time


def images_car(N):
    """
        This function creates N images of the car on the hill problem, so the position of the car and its speed. Then
        it fuse those images into a GIF.

        Parameters
        ----------
        N : int
            Number of images (steps of the car on the hill problem)

    """
    p = 0
    s = 0
    images = []
    for i in range(N):
        file_name = 'GIF_car/section3_{}.jpg'.format(i)
        # We need to add the two following lines otherwise there will be a problem to display the graph
        # because if s is bigger than the max speed which is 3, then there is an error for the color
        if p > 1:
            nice_p = 1
        elif p < -1:
            nice_p = -1
        else:
            nice_p = p

        if s > 3:
            nice_s = 3
        elif s < -3:
            nice_s = -3
        else:
            nice_s = s

        save_caronthehill_image(nice_p, nice_s, file_name)
        images.append(imageio.imread(file_name))

        u = random_policy()
        p, s = dynamics(p, s, u)

    imageio.mimsave('car_animation.gif', images)


if __name__ == "__main__":
    t = time.time()
    images_car(300)
    print("It took " + str(time.time() - t) + " seconds to generate 300 images")
