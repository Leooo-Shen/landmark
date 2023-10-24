import cv2
import os

from Preparation.util import read_points_file, recognized_text_dir, recognized_image_dir, get_point_directories, \
    save_point, show_points, sort_points

assert os.path.exists(recognized_text_dir) and os.path.exists(recognized_image_dir), "Directory does not exist"

num_recognized = 0
for subdir, dirs, files in os.walk(recognized_text_dir):
    for file in files:
        num_recognized += 1

num_recognized2 = 0
for subdir, dirs, files in os.walk(recognized_image_dir):
    for file in files:
        num_recognized2 += 1

assert num_recognized == num_recognized2, 'Problem occurs. The number of text files does not ' \
                                          'equal the number of images. There must be one text file for each image!'

average_x = 0
average_y = 0

point_name = 'S_Point'

for subdir, dirs, files in os.walk(recognized_image_dir):
    for file in files:
        curr_file_name = os.path.splitext(file)[0]
        image_path = os.path.join(recognized_image_dir, curr_file_name + '.png')
        txt_path = os.path.join(recognized_text_dir, curr_file_name + '.txt')
        image = cv2.imread(image_path, 0)
        w, h = image.shape
        all_points = read_points_file(txt_path)

        try:
            # Read N_point
            no_point_txt, _, _ = get_point_directories('No_Point')
            no_point = read_points_file(os.path.join(no_point_txt, curr_file_name + '.txt'))[0]
        except FileNotFoundError:
            cv2.imwrite('../00_Bad_Images/Unmatched_B_No_n_small' + curr_file_name + '.png', image)

        roi4_points = []
        for (x, y) in all_points:
            if y > no_point[1]:
                roi4_points.append([x, y])

        roi1_points = all_points
        roi1_points = []
        for (x, y) in all_points:
            roi1_points.append([x, y])

        if len(roi4_points) < 24:
            roi1_points = sort_points(roi1_points, by_x=True, ascending=False)[31:]
            roi1_points = sort_points(roi1_points, by_x=True, ascending=True)[2:]
            point = sort_points(roi1_points, by_x=False, ascending=True)[0]
        elif len(roi4_points) == 24:
            roi1_points = sort_points(roi1_points, by_x=True, ascending=False)[31:]
            roi1_points = sort_points(roi1_points, by_x=True, ascending=True)[2:]
            point = sort_points(roi1_points, by_x=False, ascending=True)[0]
        elif len(roi4_points) == 25:
            roi1_points = sort_points(roi1_points, by_x=True, ascending=False)[31:]
            roi1_points = sort_points(roi1_points, by_x=True, ascending=True)[2:]
            point = sort_points(roi1_points, by_x=False, ascending=True)[0]
        elif len(roi4_points) == 26:
            roi1_points = sort_points(roi1_points, by_x=True, ascending=False)[31:]
            roi1_points = sort_points(roi1_points, by_x=True, ascending=True)[2:]
            point = sort_points(roi1_points, by_x=False, ascending=True)[0]
        elif len(roi4_points) == 27:
            roi1_points = sort_points(roi1_points, by_x=True, ascending=False)[31:]
            roi1_points = sort_points(roi1_points, by_x=True, ascending=True)[2:]
            point = sort_points(roi1_points, by_x=False, ascending=True)[0]
        elif len(roi4_points) == 28:
            roi1_points = sort_points(roi1_points, by_x=True, ascending=False)[31:]
            show_points(image, roi1_points)
            roi1_points = sort_points(roi1_points, by_x=True, ascending=True)[2:]
            show_points(image, roi1_points)
            point = sort_points(roi1_points, by_x=False, ascending=True)[0]
            show_points(image, [point])
        elif len(roi4_points) == 29:
            roi1_points = sort_points(roi1_points, by_x=True, ascending=False)[31:]
            roi1_points = sort_points(roi1_points, by_x=True, ascending=True)[2:]
            point = sort_points(roi1_points, by_x=False, ascending=True)[0]
        elif len(roi4_points) == 30:
            roi1_points = sort_points(roi1_points, by_x=True, ascending=False)[31:]
            roi1_points = sort_points(roi1_points, by_x=True, ascending=True)[2:]
            point = sort_points(roi1_points, by_x=False, ascending=True)[0]
        elif len(roi4_points) == 31:
            roi1_points = sort_points(roi1_points, by_x=True, ascending=False)[31:]
            roi1_points = sort_points(roi1_points, by_x=True, ascending=True)[2:]
            point = sort_points(roi1_points, by_x=False, ascending=True)[0]
        elif len(roi4_points) == 32:
            roi1_points = sort_points(roi1_points, by_x=True, ascending=False)[31:]
            roi1_points = sort_points(roi1_points, by_x=True, ascending=True)[2:]
            point = sort_points(roi1_points, by_x=False, ascending=True)[0]
        elif len(roi4_points) == 33:
            roi1_points = sort_points(roi1_points, by_x=True, ascending=False)[31:]
            roi1_points = sort_points(roi1_points, by_x=True, ascending=True)[2:]
            point = sort_points(roi1_points, by_x=False, ascending=True)[0]
        elif len(roi4_points) > 33:
            roi1_points = sort_points(roi1_points, by_x=True, ascending=False)[31:]
            roi1_points = sort_points(roi1_points, by_x=True, ascending=True)[2:]
            point = sort_points(roi1_points, by_x=False, ascending=True)[0]
        else:
            point = None

        if point is not None:
            save_point(point_name, point, image, curr_file_name)
        else:
            cv2.imwrite('../00_Bad_Images/' + curr_file_name + '.png', image)

        average_x += point[0]
        average_y += point[1]

average_x = average_x/num_recognized
average_y = average_y/num_recognized
print('Average x coordinates: ', average_x)
print('Average y coordinates: ', average_y)

f = open('Mean_X_Y.txt', 'w')
f.write('Mean_X: ' + str(round(average_x)) + ', Mean_Y: ' + str(round(average_y)) + ';\n')
f.close()