from re import template
import numpy as np, cv2
import os,glob
import random

root_dir = 'C:/Users/LG/Desktop/12311' 
 
img_path = glob.glob(os.path.join('C:/Users/LG/Desktop/12311','*_1gacha.*')) 

template_images = ['pickupme\kotama.png','pickupme\haruka.png','pickupme\serina.png','pickupme\yuuka.png']

img_name = img_path[0]
img_array = np.fromfile(img_name, np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
img_draw = img.copy()
totalmax_val = []

def template_methods():
    th,tw = template.shape[:2]
    methods = ['cv2.TM_CCOEFF_NORMED']
    for i, method_name in enumerate(methods):
        method = eval(method_name)
        res = cv2.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        print(method_name, min_val, max_val, min_loc, max_loc)
        if method in [cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc 
            match_val = min_val 
        else:
            top_left = max_loc
            match_val = max_val 
        

    if max_val > 0.4:
        bottom_right = (top_left[0] + tw, top_left[1] + th)
        cv2.rectangle(img_draw, top_left, bottom_right, (0, 0, 255), 2)
        cv2.putText(img_draw, str(match_val), top_left,
        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1, cv2.LINE_AA)
        img_resized = cv2.resize(img_draw,(1600,200))
        cv2.imshow(method_name, img_resized)
    elif max_val < 0.4:
        print("뽑기실패")
        
    totalmax_val.append(max_val)
    print(totalmax_val)
    
   

for i in range(0,4):
    template = cv2.imread(template_images[i])
    print(str(i)+'번쨰 타겟')
    template_methods()
    

result = sum(totalmax_val)


cv2.waitKey()
cv2.destroyAllWindows()





