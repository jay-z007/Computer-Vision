import cv2
import numpy as np
from main import *

drawing = False
marked_bg_pixels = []
len_bg = 0

marked_fg_pixels = []
len_fg = 0
mode = None

def mark_seeds(event,x,y,flags,param):
	global h,w,c,img, drawing,mode,marked_bg_pixels,marked_fg_pixels
	size = 4

	if event == cv2.EVENT_LBUTTONDOWN:
		drawing = True
	elif event == cv2.EVENT_MOUSEMOVE:
		if drawing == True:
			if mode == "ob":
				if(x>=0 and x<=w-1) and (y>0 and y<=h-1):
					marked_fg_pixels.append((y,x))
				cv2.circle(img, (x,y), size, (0,0,255), -1)
			else:
				if(x>=0 and x<=w-1) and (y>0 and y<=h-1):
					marked_bg_pixels.append((y,x))
				cv2.circle(img, (x,y), size, (255,0,0), -1)
	elif event == cv2.EVENT_LBUTTONUP:
		drawing = False
		if mode == "ob":
			cv2.circle(img, (x,y), size, (0,0,255), -1)
		else:
			cv2.circle(img, (x,y), size, (255,0,0), -1)

# Create a black image, a window and bind the function to window
img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
h,w,c=img.shape
img_marking = np.zeros_like(img) 
region_size=20
cv2.namedWindow('Mark the object and background')
cv2.setMouseCallback('Mark the object and background',mark_seeds)
	

centers, colors_hists, segments, neighbor_vertices = superpixels_histograms_neighbors(img)

def make_marking():
	global img_marking, marked_fg_pixels, marked_bg_pixels
	
	for pixels in marked_fg_pixels:
		img_marking[pixels[0], pixels[1], 2] = 255

	for pixels in marked_bg_pixels:
		img_marking[pixels[0], pixels[1], 0] = 255

	return img_marking


while(1):
	
	cv2.imshow('Mark the object and background',img)
	k = cv2.waitKey(1) & 0xFF
	if k == ord('o'):
		mode = "ob"
	elif k == ord('b'):
		mode = "bg"
	elif k == 27:
		break
	
	# img_marking = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)

	# ======================================== #
	# write all your codes here
	if len_fg == len(marked_fg_pixels) and len_bg == len(marked_bg_pixels):
		continue

	img_marking = make_marking()

	fg_segments, bg_segments = find_superpixels_under_marking(img_marking, segments)

	fg_hist = cumulative_histogram_for_superpixels(fg_segments, colors_hists)
	bg_hist = cumulative_histogram_for_superpixels(bg_segments, colors_hists)

	graph_cut = do_graph_cut((fg_hist, bg_hist), (fg_segments, bg_segments), normalize_histograms(colors_hists), neighbor_vertices)

	segmask = pixels_for_segment_selection(segments, np.nonzero(graph_cut))
	segmask = np.uint8(segmask * 255)

	cv2.imshow("imgMask", segmask)
	cv2.waitKey()

	# mask = cv2.cvtColor(segmask, cv2.COLOR_BGR2GRAY) # dummy assignment for mask, change it to your result

    # ======================================== #

	# read video file
	# output_name = sys.argv[2] + "mask.png"
	# cv2.imwrite(output_name, mask);

cv2.destroyAllWindows()
