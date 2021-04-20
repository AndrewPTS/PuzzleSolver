import cv2
import glob
import numpy as np
import math
from PIL import Image

class Piece:
    points = []
    corners = []
    edges = []
    type = 0 # 0 for interior, 1 for edge, 2 for corner
    id = 0
    image = []
    mask = []

    # order: top-left clockwise
    def __init__(self, ordered_points, corners, index, image, mask):
        self.points = ordered_points
        self.corners = corners
        self.id = index
        self.image = image
        self.mask = mask
        self.edges = []

        i = 0
        side = 0
        while i < len(self.points):
            if not is_corner(self.points[(i+1)%len(self.points)], self.corners):
                first = i
                second = (i+1) % len(self.points)
                third = (i+2) % len(self.points)

                l_dist = np.linalg.norm(np.array(self.points[first])-np.array(self.points[second]))
                r_dist = np.linalg.norm(np.array(self.points[second])-np.array(self.points[third]))
                angle = calculateAngle([self.points[first], self.points[second],
                    self.points[third]])

                edge_type = 0
                if not point_in_rect(self.points[second], corners):
                    edge_type = 1

                copy = image.copy()
                cv2.circle(copy, tuple(self.points[second]), radius=10, color=(0, 255, 0), thickness=-1)
                cv2.imshow(str(i), copy)
                cv2.waitKey()
                cv2.destroyAllWindows()
                print(edge_type)

                # for a normal jigsaw piece, consider using dom_color to get the dominant color
                # in the protrusion/hole
                x, y = self.points[second]
                color = image[y][x]

                self.edges.append(Edge([l_dist, r_dist], angle, edge_type,
                    side, self.id, color))
                i += 2
            else:
                print("edge")
                i += 1
            side += 1
        print(self.edges)

class Edge:
    lengths = []
    angle = 0
    edge_type = 0    # 0 if f, 1 is m
    side = 0    # 0 is top_left, clockwise from there
    piece_id = 0
    color = 0

    def __init__(self, lengths, angle, edge_type, side, piece_id, color):
        self.lengths = lengths
        self.angle = angle
        self.edge_type = edge_type
        self.side = side
        self.piece_id = piece_id
        self.color = color

def solve(title):
    gen_edge_info(title)

def gen_edge_info(title):
    path = title + "/scrambled"
    pieces_glob = glob.glob(path + "/*")
    pieces = []
    piece_info = []
    for piece_path in pieces_glob:
        index = len(pieces)
        img = cv2.imread(piece_path)
        pieces.insert(index, img)
        piece_info.insert(index, processImg(pieces[index], index))
    print(piece_info)


def processImg(img, index):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)

    bg_color = [0, 0, 0]
    bg = np.full(img.shape, bg_color, np.uint8)
    bg_mask = cv2.bitwise_not(thresh)
    fg_masked = cv2.bitwise_and(img, img, mask=thresh)
    bg_masked = cv2.bitwise_and(bg, bg, mask=bg_mask)
    img = cv2.bitwise_or(fg_masked, bg_masked)

    #cv2.imshow("mask", img)
    #cv2.waitKey()
    #cv2.destroyAllWindows()

    # find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    eps = 0.01*cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], eps, True)

    copy = img.copy()
    cv2.drawContours(copy, [approx], 0, (200, 0, 200), 3)
    cv2.imshow("piece info", copy)
    cv2.waitKey()
    cv2.destroyAllWindows()

    points = approx[:, 0, :].tolist()
    corners = find_corners(points)
    corners = order_corners(corners)
    corners_to_draw = np.array(corners, np.int32)
    corners_to_draw = corners_to_draw.reshape((-1,1,2))
    copy = img.copy()
    cv2.polylines(copy, [corners_to_draw], True, (200, 100, 200), 3)
    cv2.imshow("corners", copy)
    cv2.waitKey()
    cv2.destroyAllWindows()
    ordered_pts = order_pts_clockwise(points, corners[0])
    cv2.circle(copy, tuple(corners[0]), radius=10, color=(255, 0, 0), thickness=-1)
    #cv2.imshow(str("ref"), copy)

    #for i in range(0, len(ordered_pts)):
    #    cv2.circle(copy, tuple(ordered_pts[i]), radius=10, color=(0,0,255), thickness=-1)
    #    cv2.imshow(str(i), copy)
    #    cv2.waitKey()
    #    cv2.destroyAllWindows()

    if len(ordered_pts) < 7:
        print("type: corner")
    elif len(ordered_pts) == 7:
        print("type: edge")
    else:
        print("type: interior")
    cv2.waitKey()
    cv2.destroyAllWindows()

    return Piece(ordered_pts, corners, index, img, thresh)

def is_right_angle(points):

    a = np.array(points[0])
    b = np.array(points[1])
    c = np.array(points[2])

    ba = a-b
    bc = c-b

    np.dot(ba, bc)
    np.linalg.norm(ba)
    np.linalg.norm(bc)
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cos_angle)

    angle_diff = abs(math.pi/2 - angle)

    #print("angle: " + str(angle) + " " + str(angle_diff))
    return angle_diff < .0075

# corner point must have a right angle in one of the following sets of points:
# [self, +1, +2], [self, +1, +3], [self, +2, +3], [self, +2, +4] (points wrap around)
# note that ALL points in the right angle are corner points, leaving one remaining one to find
def find_corners(points):

    print("STATUS: finding corners...")
    #print(points)

    if len(points) < 4:
        print("ERROR: not enough points")
        return
    corners = []

    curr = 0
    while len(corners) < 4:
        if is_right_angle([points[curr%len(points)], points[(curr+1)%len(points)],
            points[(curr+2)%len(points)]]):

            corners.insert(len(corners), points[curr%len(points)])
            corners.insert(len(corners), points[(curr+1)%len(points)])
            corners.insert(len(corners), points[(curr+2)%len(points)])

            if is_right_angle([points[(curr+1)%len(points)], points[(curr+2)%len(points)],
            points[(curr+3)%len(points)]]):
                corners.insert(len(corners), points[(curr + 3) % len(points)])
            else:
                corners.insert(len(corners), points[(curr + 4) % len(points)])

        elif is_right_angle([points[curr%len(points)], points[(curr+1)%len(points)],
            points[(curr+3)%len(points)]]):

            corners.insert(len(corners), points[curr%len(points)])
            corners.insert(len(corners), points[(curr+1)%len(points)])
            corners.insert(len(corners), points[(curr+3)%len(points)])

            if is_right_angle([points[(curr+1)%len(points)], points[(curr+3)%len(points)],
            points[(curr+4)%len(points)]]):
                corners.insert(len(corners), points[(curr + 4) % len(points)])
            else:
                corners.insert(len(corners), points[(curr + 5) % len(points)])

        elif is_right_angle([points[curr%len(points)], points[(curr+2)%len(points)],
            points[(curr+3)%len(points)]]):

            corners.insert(len(corners), points[curr%len(points)])
            corners.insert(len(corners), points[(curr+2)%len(points)])
            corners.insert(len(corners), points[(curr+3)%len(points)])

            if is_right_angle([points[(curr+2)%len(points)], points[(curr+3)%len(points)],
            points[(curr+4)%len(points)]]):
                corners.insert(len(corners), points[(curr + 4) % len(points)])
            else:
                corners.insert(len(corners), points[(curr + 5) % len(points)])

        elif is_right_angle([points[curr%len(points)], points[(curr+2)%len(points)],
            points[(curr+4)%len(points)]]):

            corners.insert(len(corners), points[curr%len(points)])
            corners.insert(len(corners), points[(curr+2)%len(points)])
            corners.insert(len(corners), points[(curr+4)%len(points)])

            if is_right_angle([points[(curr+2)%len(points)], points[(curr+4)%len(points)],
            points[(curr+5)%len(points)]]):
                corners.insert(len(corners), points[(curr + 5) % len(points)])
            else:
                corners.insert(len(corners), points[(curr + 6) % len(points)])
        else:
            curr += 1

    print("STATUS: corners found")
    return corners

def order_corners(corners):
    corners = np.array(corners)
    x_sort = corners[np.argsort(corners[:, 0]), :]

    left_pair = x_sort[:2, :]
    right_pair = x_sort[2:, :]

    left_pair = left_pair[np.argsort(left_pair[:, 1]), :]
    (top_left, bottom_left) = left_pair

    right_pair = right_pair[np.argsort(right_pair[:, 1]), :]
    (top_right, bottom_right) = right_pair

    return [top_left, top_right, bottom_right, bottom_left]

# sorts based on angle from origin point (top left corner)
def order_pts_clockwise(points, orig):

    # realign points to start from top_left corner
    # (end from top_left, then reorder)
    while not point_equals(points[len(points)-1], orig):
        points.append(points.pop(0))

    points.reverse()
    return points

def point_equals(point0, point1):
    if point0[0] == point1[0] and point0[1] == point1[1]:
        return True
    else:
        return False

def is_corner(point, corners):
    for corner in corners:
        if point_equals(point, corner):
            return True
    return False

def point_in_rect(point, corners):
    apd = triangle_area(corners[0], point, corners[3])
    dpc = triangle_area(corners[3], point, corners[2])
    cpb = triangle_area(corners[2], point, corners[1])
    pba = triangle_area(corners[1], point, corners[0])

    # print("sum: " + str(apd + dpc + cpb + pba) + " vs. rect: " + str(rect_area(corners)))
    rect = rect_area(corners)

    if (apd + dpc + cpb + pba - rect) > rect/400:
        return False
    return True

def rect_area(corners):
    w = math.hypot(corners[1][0] - corners[0][0], corners[1][1] - corners[0][1])
    h = math.hypot(corners[2][0] - corners[1][0], corners[2][1] - corners[1][1])
    return w*h

def triangle_area(a, b, c):
    l1 = math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    l2 = math.sqrt((b[0] - c[0])**2 + (b[1] - c[1])**2)
    l3 = math.sqrt((c[0] - a[0])**2 + (c[1] - a[1])**2)
    s = (l1 + l2 + l3) / 2
    area = math.sqrt(s * (s-l1) * (s-l2) * (s-l3))
    return area

# calculate the corner of angles
def calculateAngle(pts):

    a = np.array(pts[0])
    b = np.array(pts[1])
    c = np.array(pts[2])

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba,bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cos_angle)

    return angle

def dom_color(img, p_size=10):

    cv2.imshow("dom", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    pil_img = Image.fromarray(np.uint8(img))

    paletted = pil_img.convert('P', palette=Image.ADAPTIVE, colors=p_size)
    palette = paletted.getpalette()
    color_counts = sorted(paletted.getcolors(), reverse=True)
    palette_index = color_counts[0][1]
    dom_color = palette[palette_index*3:palette_index*3+3]

    if (np.array_equal(dom_color, [0,0,0])):
        palette_index = color_counts[1][1]
        dom_color = palette[palette_index*3:palette_index*3+3]

    return dom_color