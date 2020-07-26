import cv2
import argparse
import numpy as np
import pandas as pd
from imutils.video import VideoStream
from utils import detector_utils
import datetime

detection_graph, sess = detector_utils.load_inference_graph()

if __name__ == '__main__':
	# Detection confidence threshold to draw bounding box
	score_thresh  = 0.80
	#Orientation = 'bt'
	machine_border_perc = float(15)
	safety_border_perc = float(30)
	# Start the video stream
	vs = cv2.VideoCapture(0)

	cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)

	im_height, im_width = None, None
	num_hands_detect = 2
	start_time = datetime.datetime.now()
	num_frames = 0

	try:
		while True:
			rec, frame = vs.read()

			if im_height==None:
				im_height, im_width = frame.shape[:2]

			# Convert image to rgb since opencv loads images in bgr, if not accuracy will decrease.
			try:
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			except:
				print('Error converting to RGB')

			boxes, scores, classes = detector_utils.detect_objects(frame, detection_graph, sess)

			safety_position = detector_utils.draw_safety_lines(frame, machine_border_perc, safety_border_perc)
			
			detector_utils.draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, classes, im_width, im_height, frame, safety_position)

			num_frames += 1
			elasped_time = (datetime.datetime.now()-start_time).total_seconds()
			fps = num_frames/elasped_time
			cv2.putText(frame, 'FPS: '+str('%0.2f'%(fps)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

			cv2.imshow('Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

			if cv2.waitKey(1) & 0xFF==ord('q'):
				vs.release()
				cv2.destroyAllWindows()
				break
					
	except:
		pass