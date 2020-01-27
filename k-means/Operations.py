import math


def euclidean_distance(p1, p2):
    """ Calculates the distance between two data points.
    :param p1: First point.
    :param p2: Second point.
    :return: The distance between p1 and p2.
    """
    if p1.shape != p2.shape:
        raise ValueError("Dimensions must be equal.")

    dist = 0
    for i in range(p1.shape[0]):
        val = (p1[i] - p2[i])
        dist += val * val
    return math.sqrt(dist)


def closest_to_target(target, points):
    """ Finds the closest data point to the target.
    :param target: The target.
    :param points: The data points.
    :return: The index and distance of the closest point.
    """
    index = 0
    min_dist = float('inf')
    for i in range(0, points.shape[0]):
        dist = euclidean_distance(target, points[i])
        if dist < min_dist:
            min_dist = dist
            index = i
    return index, min_dist

