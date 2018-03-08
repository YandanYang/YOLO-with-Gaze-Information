# k-means ++ for YOLOv2 anchors
import numpy as np


class Box():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h



def overlap(x1, len1, x2, len2):
    len1_half = len1 / 2
    len2_half = len2 / 2

    left = max(x1 - len1_half, x2 - len2_half)
    right = min(x1 + len1_half, x2 + len2_half)

    return right - left


def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0

    area = w * h
    return area



def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u



def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)



def init_centroids(boxes,n_anchors):
    centroids = []
    boxes_num = len(boxes)

    centroid_index = np.random.choice(boxes_num, 1)
    centroids.append(boxes[centroid_index])

    print(centroids[0].w,centroids[0].h)

    for centroid_index in range(0,n_anchors-1):

        sum_distance = 0
        distance_thresh = 0
        distance_list = []
        cur_sum = 0

        for box in boxes:
            min_distance = 1
            for centroid_i, centroid in enumerate(centroids):
                distance = (1 - box_iou(box, centroid))
                if distance < min_distance:
                    min_distance = distance
            sum_distance += min_distance
            distance_list.append(min_distance)

        distance_thresh = sum_distance*np.random.random()

        for i in range(0,boxes_num):
            cur_sum += distance_list[i]
            if cur_sum > distance_thresh:
                centroids.append(boxes[i])
                print(boxes[i].w, boxes[i].h)
                break

    return centroids


def do_kmeans(n_anchors, boxes, centroids):
    loss = 0
    groups = []
    new_centroids = []
    for i in range(n_anchors):
        groups.append([])
        new_centroids.append(Box(0, 0, 0, 0))

    for box in boxes:
	#print ("box.w")	
	#print box.w #w
        min_distance = 1
        group_index = 0
        for centroid_index, centroid in enumerate(centroids):
            distance = (1 - box_iou(box, centroid))
            if distance < min_distance:
                min_distance = distance
                group_index = centroid_index
	#print group_index,len(groups[group_index])#
        groups[group_index].append(box)
        loss += min_distance
       # print len(groups[group_index])#
	
        new_centroids[group_index].w += box.w
	#print ("new_centroids[i].w")
	#print new_centroids[group_index].w	
        new_centroids[group_index].h += box.h
	#print ("new_centroids[group_index].w")
	#print new_centroids[group_index].w / len(groups[i])

    for i in range(n_anchors):
        new_centroids[i].w /= len(groups[i])
	##print len(groups[i])#
	#print ("new_centroids[i].w")
	#print new_centroids[i].w#
        new_centroids[i].h /= len(groups[i])
    return new_centroids, groups, loss


def compute_centroids(label_path,n_anchors,loss_convergence,grid_size,iterations_num,plus):

    boxes = []
    label_files = []
    f = open(label_path)
    for line in f:
        label_path = line.rstrip().replace('images', 'labels')
        label_path = label_path.replace('JPEGImages', 'labels')
        label_path = label_path.replace('.jpg', '.txt')
        label_path = label_path.replace('.JPEG', '.txt')
	label_path = label_path.replace('.png', '.txt')
        label_files.append(label_path)
    f.close()

    for label_file in label_files:
        f = open(label_file)
        for line in f:
            temp = line.strip().split(" ")
            if len(temp) > 1:
                boxes.append(Box(0, 0, float(temp[3]), float(temp[4])))

    if plus:
        centroids = init_centroids(boxes, n_anchors)
    else:
        centroid_indices = np.random.choice(len(boxes), n_anchors)
        centroids = []
        for centroid_index in centroid_indices:
            centroids.append(boxes[centroid_index])

    # iterate k-means
    centroids, groups, old_loss = do_kmeans(n_anchors, boxes, centroids)
    iterations = 1
    while (True):
        centroids, groups, loss = do_kmeans(n_anchors, boxes, centroids)
        iterations = iterations + 1
        print("loss = %f" % loss)
        if abs(old_loss - loss) < loss_convergence or iterations > iterations_num:
            break
        old_loss = loss

        for centroid in centroids:
            print(centroid.w * grid_size, centroid.h * grid_size)

    # print result
    for centroid in centroids:
        print("k-means result:\n")
        print(centroid.w * grid_size, centroid.h * grid_size)


label_path = "/home/gazetracker/otherSw/darknet/data/kitchen/kitchen_all.txt"
n_anchors = 14
loss_convergence = 1e-6
grid_size = 13
iterations_num = 100
plus = 0
compute_centroids(label_path,n_anchors,loss_convergence,grid_size,iterations_num,plus)
