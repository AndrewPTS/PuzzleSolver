import cv2
import glob
import numpy as np
import math
import sys
import traceback
from PIL import Image
from sklearn.cluster import KMeans

class Piece:
    points = []
    corners = []
    edges = []
    type = 0 # 0 for interior, 1 for edge, 2 for corner
    id = 0
    image = []
    mask = []

    # order: top-left clockwise
    def __init__(self, ordered_points, corners, index, type, image, mask):
        self.points = ordered_points
        self.corners = corners
        self.id = index
        self.image = image
        self.mask = mask
        self.edges = []
        self.type = type
        self.clockwise_rotation = 0

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
                if side % 2 == 0:
                    depth = abs(self.points[first][1] - self.points[second][1])
                else:
                    depth = abs(self.points[first][0] - self.points[second][0])

                edge_type = 0
                if not point_in_rect(self.points[second], corners):
                    edge_type = 1

                #copy = image.copy()
                #cv2.circle(copy, tuple(self.points[second]), radius=10, color=(0, 255, 0), thickness=-1)
                #cv2.imshow(str(i), copy)
                #cv2.waitKey()
                #cv2.destroyAllWindows()
                #print(edge_type)

                # for a normal jigsaw piece, consider using dom_color to get the dominant color
                # in the protrusion/hole
                x, y = self.points[second]
                color = image[y][x]

                corner_colors = [0, 0]
                x1, y1 = self.points[first]
                x2, y2 = self.points[third]
                corner_colors[0] = image[y1][x1]
                corner_colors[1] = image[y2][x2]

                self.edges.append(Edge([l_dist, r_dist], angle, edge_type,
                    side, self.id, color, corner_colors, depth))
                i += 2
            else:
                i += 1
            side += 1

    def get_edge(self, side):
        for edge in self.edges:
            if edge.side == side:
                return edge
        return None

    def get_edge_type(self, side):
        for edge in self.edges:
            if edge.side == side:
                return edge.edge_type
        return -1

    # NOTE: points and corners are not used at the stage where this is used
    #       so there is no need to rotate them
    def rotate(self, num_rotations):
        if num_rotations == 1:
            self.image = cv2.rotate(self.image, cv2.ROTATE_90_CLOCKWISE)
            self.mask = cv2.rotate(self.mask, cv2.ROTATE_90_CLOCKWISE)
        elif num_rotations == 2:
            self.image = cv2.rotate(self.image, cv2.ROTATE_180)
            self.mask = cv2.rotate(self.mask, cv2.ROTATE_180)
        elif num_rotations == 3:
            self.image = cv2.rotate(self.image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.mask = cv2.rotate(self.mask, cv2.ROTATE_90_COUNTERCLOCKWISE)

        for edge in self.edges:
            edge.side = edge.side + num_rotations
            edge.side = edge.side % 4

    def mask_image(self):
        x, y, w, h = cv2.boundingRect(self.mask)
        img = self.image[y:y+h, x:x+w]
        return img

    def offset(self, side):
        edge = self.get_edge(side)
        if edge is not None:
            return edge.depth
        return 0

class Edge:
    lengths = []
    angle = 0
    edge_type = 0    # 0 if f, 1 is m
    side = 0    # 0 is top_left, clockwise from there
    piece_id = 0
    color = 0
    corner_colors = []
    label = -1
    depth = 0

    def __init__(self, lengths, angle, edge_type, side, piece_id, color, corner_colors, depth):
        self.lengths = lengths
        self.angle = angle
        self.edge_type = edge_type
        self.side = side
        self.piece_id = piece_id
        self.color = color
        self.corner_colors = corner_colors
        self.depth = depth

    def set_label(self, label):
        self.label = label

def solve(title):
    piece_info = gen_edge_info(title)
    seed = 0
    while seed < 4:
        print("STARTING WITH SEED: " + str(seed))
        edge_set = generate_set_hierarchy(piece_info)

        solution = [[]]
        try:
            init_solution(seed, piece_info, edge_set, solution)
            row = 0
            col = 1
            while edges_remaining(edge_set) > 0:
                row, col = find_next_piece(piece_info, edge_set, solution, row, col)
        except:
            print(traceback.format_exc())
            seed += 1
            continue
        full_image = stitch_solution(solution)
        cv2.imshow("solved!", full_image)
        cv2.waitKey()
        cv2.destroyAllWindows()
        filename = title + "/" + "solution.png"
        cv2.imwrite(filename, full_image)
        return full_image
        break
    return None

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

    return piece_info

def processImg(img, index):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255-gray
    #ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    #thresh = cv2.dilate(thresh, kernel=np.ones((3,3), np.uint8))
    #thresh = cv2.erode(thresh, kernel=np.ones((1,1), np.uint8), iterations=1)

    # find contours
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    eps = 0.01 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, eps, True)
    cv2.drawContours(mask, [approx], 0, (255,255,255), -1)
    ret, thresh = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    #cv2.imshow("mask", thresh)
    #cv2.waitKey()
    #cv2.destroyAllWindows()

    bg_color = [0, 0, 0]
    bg = np.full(img.shape, bg_color, np.uint8)
    bg_mask = cv2.bitwise_not(thresh)
    fg_masked = cv2.bitwise_and(img, img, mask=thresh)
    bg_masked = cv2.bitwise_and(bg, bg, mask=bg_mask)
    img = cv2.bitwise_or(fg_masked, bg_masked)

    copy = img.copy()
    cv2.drawContours(copy, [approx], 0, (200, 0, 200), 3)
    #cv2.imshow("piece info", copy)
    #cv2.waitKey()

    points = approx[:, 0, :].tolist()
    corners = find_corners(thresh, copy, points)
    corners = order_corners(corners)

    #corners_to_draw = np.array(corners, np.int32)
    #corners_to_draw = corners_to_draw.reshape((-1,1,2))
    #copy = img.copy()
    #cv2.destroyAllWindows()
    #cv2.polylines(copy, [corners_to_draw], True, (200, 100, 200), 3)
    #cv2.imshow("corners", copy)
    #cv2.waitKey()
    #cv2.destroyAllWindows()

    adjust_angle = math.atan2(corners[1][1] - corners[0][1],
        corners[1][0] - corners[0][0])
    adjust_angle = math.degrees(adjust_angle)
    if adjust_angle > 1:
        return processImg(rotate_img(img, adjust_angle), index)

    ordered_pts = order_pts_clockwise(points, corners[0])

    #cv2.circle(copy, tuple(corners[0]), radius=10, color=(255, 0, 0), thickness=-1)
    #cv2.imshow(str("ref"), copy)

    #for i in range(0, len(ordered_pts)):
    #    cv2.circle(copy, tuple(ordered_pts[i]), radius=10, color=(0,0,255), thickness=-1)
    #    cv2.imshow(str(i), copy)
    #    cv2.waitKey()
    #    cv2.destroyAllWindows()

    type = 0
    if len(ordered_pts) < 7:
        # print("type: corner")
        type = 2
    elif len(ordered_pts) == 7:
        # print("type: edge")
        type = 1

    #cv2.imshow("mask", thresh)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    return Piece(ordered_pts, corners, index, type, img, thresh)

def rotate_img(img, angle):
    center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    res = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return res

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
    # print("angle: " + str(angle) + " " + str(angle_diff))
    return angle_diff < .02

# corner point must have a right angle in one of the following sets of points:
# [self, +1, +2], [self, +1, +3], [self, +2, +3], [self, +2, +4] (points wrap around)
# note that ALL points in the right angle are corner points, leaving one remaining one to find
def find_corners(mask, img, points):

    #print("STATUS: finding corners...")
    # print(points)

    if len(points) < 4:
        print("ERROR: not enough points")
        cv2.imshow("could not find corners", img)
        cv2.imshow("mask", mask)
        cv2.waitKey()
        sys.exit()
    corners = []

    curr = 0
    while len(corners) < 4:
        if curr >= len(points):
            print("ERROR: could not find corners")
            cv2.imshow("could not find corners", img)
            cv2.imshow("mask", mask)
            cv2.waitKey()
            sys.exit()
        corners = []
        if is_right_angle([points[curr%len(points)], points[(curr+1)%len(points)],
            points[(curr+2)%len(points)]]):

            corners.insert(len(corners), points[curr%len(points)])
            corners.insert(len(corners), points[(curr+1)%len(points)])
            corners.insert(len(corners), points[(curr+2)%len(points)])

            if is_right_angle([points[(curr+1)%len(points)], points[(curr+2)%len(points)],
            points[(curr+3)%len(points)]]):
                corners.insert(len(corners), points[(curr + 3) % len(points)])
            elif is_right_angle([points[(curr+1)%len(points)], points[(curr+2)%len(points)],
            points[(curr+4)%len(points)]]):
                corners.insert(len(corners), points[(curr + 4) % len(points)])

        elif is_right_angle([points[curr%len(points)], points[(curr+1)%len(points)],
            points[(curr+3)%len(points)]]):

            corners.insert(len(corners), points[curr%len(points)])
            corners.insert(len(corners), points[(curr+1)%len(points)])
            corners.insert(len(corners), points[(curr+3)%len(points)])

            if is_right_angle([points[(curr+1)%len(points)], points[(curr+3)%len(points)],
            points[(curr+4)%len(points)]]):
                corners.insert(len(corners), points[(curr + 4) % len(points)])
            elif is_right_angle([points[(curr+1)%len(points)], points[(curr+3)%len(points)],
            points[(curr+5)%len(points)]]):
                corners.insert(len(corners), points[(curr + 5) % len(points)])

        elif is_right_angle([points[curr%len(points)], points[(curr+2)%len(points)],
            points[(curr+3)%len(points)]]):

            corners.insert(len(corners), points[curr%len(points)])
            corners.insert(len(corners), points[(curr+2)%len(points)])
            corners.insert(len(corners), points[(curr+3)%len(points)])

            if is_right_angle([points[(curr+2)%len(points)], points[(curr+3)%len(points)],
            points[(curr+4)%len(points)]]):
                corners.insert(len(corners), points[(curr + 4) % len(points)])
            elif is_right_angle([points[(curr+2)%len(points)], points[(curr+3)%len(points)],
            points[(curr+5)%len(points)]]):
                corners.insert(len(corners), points[(curr + 5) % len(points)])

        elif is_right_angle([points[curr%len(points)], points[(curr+2)%len(points)],
            points[(curr+4)%len(points)]]):

            corners.insert(len(corners), points[curr%len(points)])
            corners.insert(len(corners), points[(curr+2)%len(points)])
            corners.insert(len(corners), points[(curr+4)%len(points)])

            if is_right_angle([points[(curr+2)%len(points)], points[(curr+4)%len(points)],
            points[(curr+5)%len(points)]]):
                corners.insert(len(corners), points[(curr + 5) % len(points)])
            elif is_right_angle([points[(curr+2)%len(points)], points[(curr+4)%len(points)],
            points[(curr+6)%len(points)]]):
                corners.insert(len(corners), points[(curr + 6) % len(points)])

        if len(corners) < 4:
            corners = []
        curr += 1

    # print("STATUS: corners found")
    # print(corners)
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

# unused in this solution, but could prove useful for real jigsaw pieces
def dom_color(img, p_size=10):

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

def generate_set_hierarchy(piece_info):

    pixel_list = []
    for piece in piece_info:
        for edge in piece.edges:
            pixel_list.append(edge.color)

    clusters = math.floor(len(pixel_list)/10)
    clt = KMeans(n_clusters=clusters)
    clt.fit(pixel_list)

    f_set = []
    m_set = []
    for j in range(3):
        f_set.append([])
        m_set.append([])
        for k in range(clusters):
            f_set[j].append([])
            m_set[j].append([])
    edge_set = [f_set, m_set]

    i = 0
    for piece in piece_info:
        for edge in piece.edges:
            edge.set_label(clt.labels_[i])
            edge_set[edge.edge_type][piece.type][clt.labels_[i]].append(edge)
            sorted(edge_set[edge.edge_type][piece.type][clt.labels_[i]],
                   key=lambda x: x.angle)
            i += 1
    return edge_set

def remove_piece(piece_info, edge_set, piece_id):

    # get all edges that need to be removed
    piece = piece_info[piece_id]
    edges_to_remove = piece.edges

    for i in range(len(edges_to_remove)):
        edge = edges_to_remove[i]
        edge_set[edge.edge_type][piece.type][edge.label].remove(edge)

def init_solution(seed, piece_info, edge_set, solution):
    found = 0
    start_corner = None

    for piece in piece_info:
        if piece.type == 2:
            if found == seed:
                start_corner = piece
                break
            else:
                found += 1

    remove_piece(piece_info, edge_set, start_corner.id)

    first_edge = -1
    second_edge = -1
    for i in range(4):
        if start_corner.get_edge(i) is None:
            if first_edge == -1:
                first_edge = i
            else:
                second_edge = i
    if second_edge - first_edge == 1:
        start_corner.rotate(4 - second_edge)

    solution[0].append(start_corner)
    #cv2.imshow("start corner", start_corner.image)
    #cv2.waitKey()
    #cv2.destroyAllWindows()

def find_next_piece(piece_info, edge_set, solution, row, col):

    thresh = 0.0
    max_score = 0.0
    max_piece = None
    max_edge = None

    if row == 0:
        # only check edge and corner sets
        # top row, only match right/left edge
        left_edge = solution[0][col-1].get_edge(1)

        #cv2.imshow("left", solution[0][col-1].image)

        m = abs(left_edge.edge_type - 1)
        label = left_edge.label


        #for i in range(1,3):
        #    for b in range(len(edge_set[m][i])):
        #        n = len(edge_set[m][i][b])
        #        for j in range(n):
        #            edge = edge_set[m][i][b][j]
        #            score = edge_score(left_edge, edge)
        #            piece = piece_info[edge.piece_id]
        #            if score > -20:
        #                cv2.imshow(str(b) + " " + str(edge.side) + ": " +str(score), piece.image)
        #cv2.waitKey()
        #cv2.destroyAllWindows()

        for i in range(1, 3):
            n = len(edge_set[m][i][label])
            for j in range(n):
                edge = edge_set[m][i][label][j]
                piece = piece_info[edge.piece_id]
                score = edge_score(left_edge, edge)
                if score > thresh:
                    edge2 = piece.get_edge((edge.side + 1) % 4)
                    if edge2 is None:
                        if score > max_score:
                            max_score = score
                            max_piece = piece
                            max_edge = edge

        if max_piece is None or max_score < 50:
            # repeat, but this time search in other label bins
            for i in range(3):
                for k in range(len(edge_set[m][i])):
                    # skip label that has already been searched
                    if k == label:
                        k += 1
                        if k == len(edge_set[m][i]):
                            break
                    n = len(edge_set[m][i][k])
                    for j in range(n):
                        edge = edge_set[m][i][k][j]
                        piece = piece_info[edge.piece_id]
                        score = edge_score(left_edge, edge)
                        if score > thresh:
                            edge2 = piece.get_edge((edge.side + 1) % 4)
                            if edge2 is None:
                                if score > max_score:
                                    max_score = score
                                    max_piece = piece
                                    max_edge = edge
        #print("NEXT: " + str(max_piece))
        next_piece = max_piece
        next_piece.rotate(abs(3 - max_edge.side))
        #cv2.imshow("NEXT", next_piece.image)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        if next_piece.get_edge((max_edge.side + 2) % 4) is None:
            next_row = row + 1
            next_col = 0
        else:
            next_row = row
            next_col = col + 1

    elif col == 0:

        #cv2.imshow("top", solution[row-1][col].image)

        # only check edge and corner sets
        # left col only, only match bottom/top edge
        bot_edge = solution[row-1][0].get_edge(2)
        m = abs(bot_edge.edge_type - 1)

        #for i in range(1,3):
        #    for b in range(len(edge_set[m][i])):
        #        n = len(edge_set[m][i][b])
        #        for j in range(n):
        #            edge = edge_set[m][i][b][j]
        #            score = edge_score(bot_edge, edge)
        #            piece = piece_info[edge.piece_id]
        #            if score > -20:
        #                cv2.imshow(str(edge.side) + ": " + str(score), piece.image)
        #cv2.waitKey()
        #cv2.destroyAllWindows()


        label = bot_edge.label
        for i in range(1, 3):
            n = len(edge_set[m][i][label])
            for j in range(n):
                edge = edge_set[m][i][label][j]
                piece = piece_info[edge.piece_id]
                score = edge_score(bot_edge, edge)
                if score > thresh:
                    edge2 = piece.get_edge((edge.side + 3) % 4)
                    if edge2 is None:
                        if score > max_score:
                            max_score = score
                            max_piece = piece
                            max_edge = edge

        if max_piece is None or max_score < 50:
            # repeat, but this time search in other label bins
            for i in range(3):
                for k in range(len(edge_set[m][i])):
                    if k == label:
                        k += 1
                        if k == len(edge_set[m][i]):
                            break
                    n = len(edge_set[m][i][k])
                    for j in range(n):
                        edge = edge_set[m][i][k][j]
                        piece = piece_info[edge.piece_id]
                        score = edge_score(bot_edge, edge)
                        if score > thresh:
                            edge2 = piece.get_edge((edge.side + 3) % 4)
                            if edge2 is None:
                                if score > max_score:
                                    max_score = score
                                    max_piece = piece
                                    max_edge = edge
        next_piece = max_piece
        next_piece.rotate(abs(4 - max_edge.side))
        #cv2.imshow("NEXT", next_piece.image)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        if next_piece.get_edge((max_edge.side + 1) % 4) is None:
            next_row = row + 1
            next_col = 0
        else:
            next_row = row
            next_col = col + 1

    else:

        #cv2.imshow("left", solution[row][col-1].image)

        left_edge = solution[row][col - 1].get_edge(1)
        bot_edge = solution[row-1][col].get_edge(2)
        # match right/left edge, then verify it fits with top edge
        m = abs(left_edge.edge_type - 1)

        #for i in range(0,3):
        #    for b in range(len(edge_set[m][i])):
        #        n = len(edge_set[m][i][b])
        #        for j in range(n):
        #            edge = edge_set[m][i][b][j]
        #            score = edge_score(left_edge, edge)
        #            piece = piece_info[edge.piece_id]
        #            if score > -20:
        #                cv2.imshow(str(edge.side) + ": " +str(score), piece.image)
        #cv2.waitKey()
        #cv2.destroyAllWindows()

        label = left_edge.label
        for i in range(3):
            n = len(edge_set[m][i][label])
            for j in range(n):
                edge = edge_set[m][i][label][j]
                piece = piece_info[edge.piece_id]
                score = edge_score(left_edge, edge)
                if score > thresh:
                    edge2 = piece.get_edge((edge.side+1) % 4)
                    if edge2 is None or edge_score(bot_edge, edge2) > thresh:
                        if score > max_score:
                            max_score = score
                            max_piece = piece
                            max_edge = edge

        if max_piece is None or max_score < 50:
            # repeat, but this time look in other label bins
            for i in range(3):
                for k in range(len(edge_set[m][i])):
                    if k == label:
                        k += 1
                        if k == len(edge_set[m][i]):
                            break
                    n = len(edge_set[m][i][k])
                    for j in range(n):
                        edge = edge_set[m][i][k][j]
                        piece = piece_info[edge.piece_id]
                        score = edge_score(left_edge, edge)
                        if score > thresh:
                            edge2 = piece.get_edge((edge.side + 1) % 4)
                            if edge2 is None or edge_score(bot_edge, edge2) > thresh:
                                if score > max_score:
                                    max_score = score
                                    max_piece = piece
                                    max_edge = edge
        next_piece = max_piece
        next_piece.rotate(abs(3 - max_edge.side))
        #cv2.imshow("NEXT", next_piece.image)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        if next_piece.get_edge((max_edge.side + 2) % 4) is None:
            next_row = row + 1
            next_col = 0
        else:
            next_row = row
            next_col = col + 1

    remove_piece(piece_info, edge_set, next_piece.id)
    if row < len(solution):
        solution[row].append(next_piece)
    else:
        new_row = [next_piece]
        solution.append(new_row)
    return next_row, next_col

def color_score(rgb1, rgb2):
    b_score = float(abs(float(rgb1[0]) - float(rgb2[0])))
    g_score = float(abs(float(rgb1[1]) - float(rgb2[1])))
    r_score = float(abs(float(rgb1[2]) - float(rgb2[2])))
    return ((b_score + g_score + r_score))**1.2

def edge_score(edge1, edge2):
    angle_score = abs(math.degrees(edge1.angle) - math.degrees(edge2.angle))
    length1_score = abs(edge1.lengths[0] - edge2.lengths[1])
    length2_score = abs(edge1.lengths[1] - edge2.lengths[0])
    depth_score = abs(edge1.depth - edge2.depth)
    c_score = (color_score(edge1.color, edge2.color)/4) + \
              color_score(edge1.corner_colors[0], edge2.corner_colors[1]) \
              + color_score(edge1.corner_colors[1], edge2.corner_colors[0]) / 20

    if length1_score > 10:
        length1_score = length1_score*2
    if length2_score > 10:
        length2_score = length2_score*2
    if angle_score > 10:
        angle_score = angle_score*2
    if depth_score > 5:
        depth_score = depth_score*2
    if angle_score < 2 or (length1_score < 2 and length2_score < 2) or depth_score < 2:
        c_score = c_score/10
    score = 100 - (angle_score + length1_score + length2_score + depth_score + c_score)
    #if score > -20:
    #    print("left: " + str(edge1.corner_colors[0]) + " " + str(edge1.color) + " "
    #          + str(edge1.corner_colors[1]))
    #    print("right: " + str(edge2.corner_colors[0]) + " " + str(edge2.color) + " "
    #          + str(edge2.corner_colors[1]))
    #    print("score analysis: " + str(score) + "\n" + "angle: " + str(angle_score) + "\n" +
    #          "length1: " + str(length1_score) + "\nlength2: " + str(length2_score) + "\n" +
    #          "depth: " + str(depth_score) + "\ncolor: " + str(c_score))
    return score

def edges_remaining(edge_set):
    rem = 0

    for m in range(2):
        for t in range(3):
            for b in range(len(edge_set[m][t])):
                rem += len(edge_set[m][t][b])

    return rem

def stitch_solution(solution):
    row = 0
    img = None
    while row < len(solution):
        strip = solution[row][0].mask_image()
        max_offset = 0
        col = 1
        while col < len(solution[0]):
            next_img = solution[row][col].mask_image()
            offset = solution[row][col].offset(3)
            left_height_buffer = 0
            left_align_buffer = 0
            right_height_buffer = 0
            right_align_buffer = 0

            if col > 0:
                left_height_offset = solution[row][col-1].offset(0)
                if solution[row][col-1].get_edge_type(0) == 0:
                    left_height_offset = 0
                max_offset = max(max_offset, left_height_offset)
                left_height_offset = max_offset
                right_height_offset = solution[row][col].offset(0)
                if solution[row][col].get_edge_type(0) == 0:
                    right_height_offset = 0
                offset_difference = left_height_offset - right_height_offset
                if offset_difference > 0:
                    right_align_buffer = offset_difference
                elif offset_difference < 0:
                    left_align_buffer = abs(offset_difference)
            left_img = cv2.copyMakeBorder(strip, left_align_buffer, 0,
                                          0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            right_img = cv2.copyMakeBorder(next_img, right_align_buffer, 0,
                                           0, 0, cv2.BORDER_CONSTANT,
                                           value=[0, 0, 0])

            h, w1, _ = left_img.shape
            h2, w2, _ = right_img.shape

            if h > h2:
                right_height_buffer = h-h2
            elif h2 > h:
                left_height_buffer = h2-h

            left_img = cv2.copyMakeBorder(left_img, 0, left_height_buffer,
                    0, w2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            right_img = cv2.copyMakeBorder(right_img, 0, right_height_buffer,
                    w1 - offset, offset, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            strip = cv2.bitwise_or(left_img, right_img)
            height, width, _ = strip.shape
            strip = strip[0:height, 0:width-offset]
            col += 1
        if img is None:
            img = strip
        else:
            next_img = strip
            max_down_offset = 0
            max_up_offset = 0
            for p in range(0, len(solution[row])):
                new_offset = solution[row][p].offset(0)
                if solution[row][p].get_edge_type(0) == 0:
                    if new_offset > max_down_offset:
                        max_down_offset = new_offset
                else:
                    if new_offset > max_up_offset:
                        max_up_offset = new_offset
            offset = max_up_offset + max_down_offset
            h1, w1, _ = img.shape
            h2, w2, _ = next_img.shape

            top_side_buffer = 0
            bottom_side_buffer = 0
            if w1 > w2:
                bottom_side_buffer = w1-w2
            elif w2 > w1:
                top_side_buffer = w2-w1
            top_img = cv2.copyMakeBorder(img, 0, h2, 0, top_side_buffer, cv2.BORDER_CONSTANT,
                                          value=[0, 0, 0])
            bot_img = cv2.copyMakeBorder(next_img, h1 - offset, offset, 0, bottom_side_buffer,
                                           cv2.BORDER_CONSTANT, value=[0, 0, 0])

            img = cv2.bitwise_or(top_img, bot_img)
            height, width, _ = img.shape
            img = img[0:height - offset, 0:width]
        row += 1
    return img