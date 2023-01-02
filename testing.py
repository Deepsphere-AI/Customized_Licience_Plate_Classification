import cv2
from PIL import Image
import glob

path = "C:/Users/ds_007/anaconda3/envs/web_scrapping/Images(2)" + "/*.*"

all_imgs = []
i = 1
all_imgs = glob.glob(path,recursive=True)
print(len(all_imgs))
for image_path in glob.glob(path,recursive=True):
    try:
        image_path = image_path.replace("\\","/")
        image = cv2.imread(image_path)
        # cv2.imshow("Image",image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(f"C:/Users/ds_007/anaconda3/envs/web_scrapping/Images(3)/image{i}.png",image)
        i += 1
    except Exception as e:
        #print("The error is: ",e)
        continue

        
