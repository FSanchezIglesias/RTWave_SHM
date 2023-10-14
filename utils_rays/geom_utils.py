import numpy as np


def dot_2d(a, b):
    return a[0]*b[0] + a[1]*b[1]


def norm_2d(a):
    """
    Returns norm of vector a
    :param a: vector n components
    :return: Norm of a
    """
    # return (sum([r**2 for r in a]))**0.5
    return (dot_2d(a, a))**0.5


def cross(a, b):
    """
    Cross product of a, b in 3D
    :param a: vector 3 components
    :param b: vector 3 components
    :return: cross product, ndarray 3 components
    """
    return np.array([a[1]*b[2] - a[2]*b[1],
                     a[2]*b[0] - a[0]*b[2],
                     a[0]*b[1] - a[1]*b[0]])


def cross_2d(a, b):
    """
    Cross product of a, b in 2D
    :param a: vector 2 components
    :param b: vector 2 components
    :return: cross product, float
    """
    return a[0]*b[1] - a[1]*b[0]


def seg_seg_intersect_2d(a1, a2, b1, b2, tol=1.e-3):
    """
    Returns the point of intersection of the segments defined by a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    :return: intersection point
    """

    s = np.vstack([a1, a2, b1, b2])  # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = cross(h[0], h[1])  # get first line
    l2 = cross(h[2], h[3])  # get second line
    x, y, z = cross(l1, l2)  # point of intersection

    if z == 0:  # lines are parallel
        return None

    i = np.array((x / z, y / z))
    # Check if inside segment a1, a2 ...
    na = norm_2d(a2 - a1)
    pa = dot_2d((a2-a1)/na, i-a1)

    # Check if inside segment b1, b2 ...
    nb = norm_2d(b2 - b1)
    pb = dot_2d((b2-b1)/nb, i-b1)
    
    # chek all in one!!!
    if ((pb - nb < tol) and (pb > tol)) and ((pa - na < tol) and (pa > tol)):
        return i    
    else:
        return None
    

def circunf_seg_intersect_2d(circle_center, circle_radius, pt1, pt2,
                             full_line=False, tangent_tol=1e-9):
    """ Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.

    :param circle_center: The (x, y) location of the circle center
    :param circle_radius: The radius of the circle
    :param pt1: The (x, y) location of the first point of the segment
    :param pt2: The (x, y) location of the second point of the segment
    :param full_line: True to find intersections along full line - not just in the segment.
                      False will just return intersections within the segment.
    :param tangent_tol: Num tolerance at which we decide the intersections are close enough to consider it a tangent
    :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the
           circle intercepts a line segment.

    Note: reference: http://mathworld.wolfram.com/Circle-LineIntersection.html
    """

    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx ** 2 + dy ** 2)**.5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2

    if discriminant < 0:  # No intersection between circle and line
        return []
    else:  # There may be 0, 1, or 2 intersections with the segment
        intersections = [
            (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2,
             cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2)
            for sign in ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct

        # If only considering the segment, filter out intersections that do not fall within the segment
        if not full_line:
            fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
            intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]

        # If line is tangent to circle, return just one point (as both intersections have same location)
        if len(intersections) == 2 and abs(discriminant) <= tangent_tol:
            return [intersections[0]]
        else:
            return intersections
