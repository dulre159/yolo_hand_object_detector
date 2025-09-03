import os
from PIL import Image
import numpy as np
import math
import xml.etree.ElementTree as ET

def _is_not_legimate(ele):
	return (ele == None or ele.text == 'None' or ele.text == None)
	
def _load_pascal_annotation(anno_file_path, image_w, image_h):
	"""
	Load image and bounding boxes info from XML file in the PASCAL VOC
	format.
	"""
	filename = anno_file_path
	tree = ET.parse(filename)
	objs = tree.findall('object')
	size = tree.find('size')
	w = int(size.find('width').text)
	h = int(size.find('height').text)
	
	if image_w != w or image_h != h:
		print(f"Annotation image size diffrent then real image size - RWH: {str(image_w)}, {str(image_h)} AWH: {str(w)}, {str(h)}")
	
	out_lines = []

	# Load object bounding boxes into a data frame.
	for ix, obj in enumerate(objs):
		output_line = ""
		
		cls = obj.find('name').text.lower().strip()
		clsid = 0 if cls == "targetobject" else 1

		bbox = obj.find('bndbox')
		# Make pixel indexes 0-based
		x1 = max(float(bbox.find('xmin').text) - 1,0)
		y1 = max(float(bbox.find('ymin').text) - 1,0)
		x2 = max(float(bbox.find('xmax').text) - 1,0)
		y2 = max(float(bbox.find('ymax').text) - 1,0)
		
		# Transform the bbox co-ordinates as per the format required by YOLO v5
		b_center_x = (x1 + x2) / 2 
		b_center_y = (y1 + y2) / 2
		b_width    = (x2 - x1)
		b_height   = (y2 - y1)
		
		b_center_x /= image_w 
		b_center_y /= image_h 
		b_width    /= image_w 
		b_height   /= image_h 

					
		out_lines.append(f"{str(clsid)} {str(b_center_x)} {str(b_center_y)} {str(b_width)} {str(b_height)}")
		
	return out_lines

	
if __name__ == '__main__':

	imageset_path = './ImageSets/Main/test.txt'
	imageset_out = './valid.txt'
	xml_annos_path = './Annotations'
	images_dir_path = './JPEGImages'
	yolo_ann_output_dir = './labels' 
	
	# Create the output directory if it doesn't exist
	if not os.path.exists(yolo_ann_output_dir):
	    os.makedirs(yolo_ann_output_dir)
	
	files = []
	with open(imageset_path, "r") as file:
		files = file.readlines()
		
	handsincwassv_tot = 0
	
	outfilenames = []
	
	for filename in files:
		image_path = os.path.join(images_dir_path, filename.rstrip() + ".jpg")
		xml_anno_path = os.path.join(xml_annos_path, filename.rstrip() + ".xml")
		out_filename = "100DOH_" + filename.rstrip() + ".txt"
		out_filename_im = "100DOH_" + filename.rstrip() + ".jpg"
		out_yolo_label_path = os.path.join(yolo_ann_output_dir, out_filename)
		
		im = Image.open(image_path, mode='r')
		w, h = im.size
			
		out_lines = _load_pascal_annotation(xml_anno_path, w, h)
		
		if len(out_lines) > 0:
		
			# Write to the TXT file
			with open(out_yolo_label_path, 'w') as yolo_anno:
				for line in out_lines:
					yolo_anno.write(line + "\n")
			outfilenames.append("datasets/100doh_org/images/"+out_filename_im)
	
	with open(imageset_out, 'w') as out_imageset:
		for line in outfilenames:
			out_imageset.write(line + "\n")
		
