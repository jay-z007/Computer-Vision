import cv2
import numpy as np
from main import *

drawing = False
mode = None


def mark_seeds(event, x, y, flags, param):
    global h, w, c, img, img_marking, drawing, mode
    size = 4

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == "ob":
                cv2.circle(img, (x, y), size, (0, 0, 255), -1)
                cv2.circle(img_marking, (x, y), size, (0, 0, 255), -1)
            else:
                cv2.circle(img, (x, y), size, (255, 0, 0), -1)
                cv2.circle(img_marking, (x, y), size, (255, 0, 0), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == "ob":
            cv2.circle(img, (x, y), size, (0, 0, 255), -1)
            cv2.circle(img_marking, (x, y), size, (0, 0, 255), -1)
        else:
            cv2.circle(img, (x, y), size, (255, 0, 0), -1)
            cv2.circle(img_marking, (x, y), size, (255, 0, 0), -1)


# Create a white image, a window and bind the function to window
img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
h, w, c = img.shape
img_marking = np.ones_like(img) * 255

cv2.namedWindow('Mark the object and background')
cv2.setMouseCallback('Mark the object and background', mark_seeds)

centers, colors_hists, segments, neighbor_vertices = superpixels_histograms_neighbors(
    img)

while (1):

    cv2.imshow('Mark the object and background', img_marking)
    cv2.imshow('Mark the object and background', img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('o'):
        mode = "ob"
    elif k == ord('b'):
        mode = "bg"
    elif k == ord('g'):

        cv2.imshow("img_marking", img_marking)
        cv2.waitKey(1)

        fg_segments, bg_segments = find_superpixels_under_marking(
            img_marking, segments)

        fg_hist = cumulative_histogram_for_superpixels(fg_segments,
                                                       colors_hists)
        bg_hist = cumulative_histogram_for_superpixels(bg_segments,
                                                       colors_hists)

        graph_cut = do_graph_cut((fg_hist, bg_hist), (fg_segments, bg_segments),
                                 normalize_histograms(colors_hists),
                                 neighbor_vertices)

        segmask = pixels_for_segment_selection(segments, np.nonzero(graph_cut))
        segmask = np.uint8(segmask * 255)

        cv2.imshow("imgMask", segmask)
        cv2.waitKey()
        mask = segmask

        img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)

        img_marking = np.ones_like(img) * 255

    elif k == 27:
        break

output_name = sys.argv[2] + "mask_bonus.png"
cv2.imwrite(output_name, mask)

cv2.destroyAllWindows()
