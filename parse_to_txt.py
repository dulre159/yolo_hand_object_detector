import os
from PIL import Image
import numpy as np
import math
import xml.etree.ElementTree as ET

def _is_not_legimate(ele):
	return (ele == None or ele.text == 'None' or ele.text == None)
	
#casi che vanno bene:
#se una mano ha contactstate 3 o 4 e ha la bounding box dell'oggetto relativo
#se una mano ha contactstate 0
#se una mano ha contactstate 3 o 4 ma non ha la bounding box dell'oggetto relativo 

#Ci sono mani in contatto che non hanno il vettore associativo valido? Si 12723 di cui 12716 non hanno la bounding box del oggetto relativo valida
#Ci sono casi in cui la mano è in contatto e la magnitudo del vettore unitario non è 1 o il magnitudo == -1 o 0.0

# Mano destra == 1

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
		
		if clsid == 1:

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

			#diffc = obj.find('difficult')
			#difficult = 0 if diffc == None else int(diffc.text)
			#ishards[ix] = difficult
			
			objxmin = obj.find('objxmin')
			objxmin = -1 if _is_not_legimate(objxmin) else max(float(objxmin.text) - 1,0)
			
			objymin = obj.find('objymin')
			objymin = -1 if _is_not_legimate(objymin) else max(float(objymin.text) - 1,0)
			
			objxmax = obj.find('objxmax')
			objxmax = -1 if _is_not_legimate(objxmax) else max(float(objxmax.text) - 1,0)

			objymax = obj.find('objymax')
			objymax = -1 if _is_not_legimate(objymax) else max(float(objymax.text) - 1,0)
				
			hs = obj.find('contactstate')
			hs = -1 if _is_not_legimate(hs) else int(hs.text)

			#contactr = obj.find('contactright')
			#contactr = -1 if _is_not_legimate(contactr) else int(contactr.text)

			#contactl = obj.find('contactleft')
			#contactl = -1 if _is_not_legimate(contactl) else int(contactl.text)

			mag = obj.find('magnitude')
			mag = -1 if _is_not_legimate(mag) else float(mag.text) * 0.001 # balance scale

			dx = obj.find('unitdx')
			dx = -1 if _is_not_legimate(dx) else float(dx.text)

			dy = obj.find('unitdy')
			dy = -1 if _is_not_legimate(dy) else float(dy.text)

			lr = obj.find('handside')
			lr = 0 if _is_not_legimate(lr) else float(lr.text)
			
			uvmag = np.linalg.norm(np.array([dx,dy]))
			uvmag_close_to_one = math.isclose(uvmag,1.0, rel_tol=0.001)
			
			if hs != 3 and hs != 4:
				hs = 0
				objxmin = objymin = objxmax = objymax = mag = dx = dy = -1
				out_lines.append(f"{str(1)} {str(b_center_x)} {str(b_center_y)} {str(b_width)} {str(b_height)} {str(hs)} {str(dx)} {str(dy)} {str(mag)} {str(lr)}")
				
			if hs == 3 or hs == 4:
				hs = 1
				
				if objxmin >= 0 and objymin >= 0 and objxmax >= 0 and objymax >= 0:
					if objxmin < objxmax and objxmax < image_w and objymin < objymax and objymax < image_h:
						# Transform the obj bbox co-ordinates as per the format required by YOLO v5
						ob_center_x = (objxmin + objxmax) / 2 
						ob_center_y = (objymin + objymax) / 2
						ob_width    = (objxmax - objxmin)
						ob_height   = (objymax - objymin)
						
						ob_center_x /= image_w 
						ob_center_y /= image_h 
						ob_width    /= image_w 
						ob_height   /= image_h
						out_lines.append(f"{str(0)} {str(ob_center_x)} {str(ob_center_y)} {str(ob_width)} {str(ob_height)} {str(-1)} {str(-1)} {str(-1)} {str(-1)} {str(-1)}")
						
				out_lines.append(f"{str(1)} {str(b_center_x)} {str(b_center_y)} {str(b_width)} {str(b_height)} {str(hs)} {str(dx)} {str(dy)} {str(mag)} {str(lr)}")
		
	return out_lines

def _load_pascal_annotation_and_count_hands_in_contact_without_assov(anno_file_path):
	"""
	Load image and bounding boxes info from XML file in the PASCAL VOC
	format.
	"""
	filename = anno_file_path
	tree = ET.parse(filename)
	objs = tree.findall('object')
	
	
	handsincwassv = 0

	# Load object bounding boxes into a data frame.
	for ix, obj in enumerate(objs):
	
		cls = obj.find('name').text.lower().strip()
		clsid = 0 if cls == "targetobject" else 1
		
		if clsid == 1:

			bbox = obj.find('bndbox')
			# Make pixel indexes 0-based
			x1 = max(float(bbox.find('xmin').text) - 1,0)
			y1 = max(float(bbox.find('ymin').text) - 1,0)
			x2 = max(float(bbox.find('xmax').text) - 1,0)
			y2 = max(float(bbox.find('ymax').text) - 1,0)
			
			objxmin = obj.find('objxmin')
			objxmin = -1 if _is_not_legimate(objxmin) else int(objxmin.text)
			
			objymin = obj.find('objymin')
			objymin = -1 if _is_not_legimate(objymin) else int(objymin.text)
			
			objxmax = obj.find('objxmax')
			objxmax = -1 if _is_not_legimate(objxmax) else int(objxmax.text)

			objymax = obj.find('objymax')
			objymax = -1 if _is_not_legimate(objymax) else int(objymax.text)

			hs = obj.find('contactstate')
			hs = -1 if _is_not_legimate(hs) else int(hs.text)

			contactr = obj.find('contactright')
			contactr = -1 if _is_not_legimate(contactr) else int(contactr.text)

			contactl = obj.find('contactleft')
			contactl = -1 if _is_not_legimate(contactl) else int(contactl.text)

			mag = obj.find('magnitude')
			mag = -1 if _is_not_legimate(mag) else float(mag.text) #* 0.001 # balance scale

			dx = obj.find('unitdx')
			dx = -1 if _is_not_legimate(dx) else float(dx.text)

			dy = obj.find('unitdy')
			dy = -1 if _is_not_legimate(dy) else float(dy.text)

			lr = obj.find('handside')
			lr = -1 if _is_not_legimate(lr) else float(lr.text)
			
			uvmag = np.linalg.norm(np.array([dx,dy]))
			uvmag_close_to_zero = math.isclose(uvmag,1.0, rel_tol=0.001)
			
			#if x1 >= 0 and x2 >= 0 and y1 >= 0 and y2 >= 0:
				#if x1 > x2 or y1 > y2:
					#print("Invalid bbox")
					
			if x2 == 0 or y2 == 0:
				print("Invalid bbox")
			
			#if hs > 0 and (mag != -1 or uvmag_close_to_zero == True):
				#if objxmin == -1 or objymin == -1 or objxmax == -1 or objymax == -1:
					#print(f"objxmin: {str(objxmin)} objymin: {str(objymin)} objxmax: {str(objxmax)} objymax: {str(objymax)}")
				#print(f"hs: {str(hs)} dx: {str(dx)} dy: {str(dy)} mag: {str(mag)}") 
					#handsincwassv+=1
			
			#if hs > 0 and mag != -1 and uvmag_close_to_zero == False:
				#print(f"hs: {str(hs)} dx: {str(dx)} dy: {str(dy)} mag: {str(mag)}") 
				#handsincwassv+=1
				
			
	return handsincwassv
	
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
			outfilenames.append("erikyolov3/data/dl/images/"+out_filename_im)
	
	with open(imageset_out, 'w') as out_imageset:
		for line in outfilenames:
			out_imageset.write(line + "\n")
		
