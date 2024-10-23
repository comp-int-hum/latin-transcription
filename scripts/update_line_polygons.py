import xml.etree.ElementTree as ET
import numpy as np
import glob
import cv2
import os.path
import re
import os.path
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="source")
    parser.add_argument("--output_dir", type=str, default="modified_line_polygons")
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    def get_line_spacing(baselines):
        #print(baselines)
        center_x = np.median([(l.T[0][0] + l.T[0][-1])/2 for l in baselines])
        
        center_ys = []
        for line in baselines:
            xs, ys = line.T
            center_ys.append(np.interp(center_x, xs, ys))

        med_spacing = np.median(np.diff(center_ys))
        return int(round(med_spacing))

    def box_from_baseline(baseline_points_str, med_spacing, height):
        lower_spacing = int(round(0.23 * med_spacing - 1))
        upper_spacing = int(round(0.77 * med_spacing - 1))
        
        baseline_points = np.array([p.split(",") for p in baseline_points_str.split(" ")], dtype=int)
        
        box_points = ""
        for i in range(len(baseline_points)):
            box_points += "{},{} ".format(baseline_points[i][0], min(height-1, baseline_points[i][1] + lower_spacing))
            
        for i in range(len(baseline_points)-1, -1, -1):
            box_points += "{},{} ".format(baseline_points[i][0], max(0, baseline_points[i][1] - upper_spacing))

        return box_points[:-1] #trim whitespace

    def get_namespace(element):
        #rint(element.tag)
        m = re.match('\{.*\}', element.tag)
        return m.group(0)[1:-1] if m else ''    
    
    dirname = args.input_dir + '/'
    for filename in glob.glob(dirname + "*.xml"):
        print(filename)
        baselines = []
        tree = ET.parse(filename)
        root = tree.getroot()
        ns = {"ns": get_namespace(tree.getroot())}
        #ns = {'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'}
        ET.register_namespace('', ns['ns'])

        image_filename = root.find('ns:Page', ns).get('imageFilename')
        height = cv2.imread(dirname + image_filename, 0).shape[0]
        
        #First iteration: calculate average line spacing
        for text_region in root.findall('.//ns:TextRegion', ns):
            for lineno, text_line in enumerate(text_region.findall('.//ns:TextLine', ns)):
                baseline = text_line.find('ns:Baseline', ns).get('points')
                baselines.append(np.array([p.split(",") for p in baseline.split(" ")], dtype=int))
            
        med_spacing = get_line_spacing(baselines)

        #Second iteration: update bounding boxes
        for text_region in root.findall('.//ns:TextRegion', ns):
            for lineno, text_line in enumerate(text_region.findall('.//ns:TextLine', ns)):
                baseline = text_line.find('ns:Baseline', ns).get('points')          
                updated_box_text = box_from_baseline(baseline, med_spacing, height)
                box = text_line.find('ns:Coords', ns)
                box.set('points', updated_box_text)

        tree.write(args.output_dir + '/' + os.path.basename(filename))