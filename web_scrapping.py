# #"C:/Users/ds_007/anaconda3/envs/web_scrapping/chromedriver_win32/chromedriver.exe"

# from selenium import webdriver
# from pynput.keyboard import Key,Controller
# from selenium.webdriver.common.keys import Keys
# import time
# from selenium.webdriver.common.by import By
# import requests
# import io
# from PIL import Image
# import cv2
# import os

# path = 'C://Users//ds_007/anaconda3//envs//web_scrapping//chromedriver_win32//chromedriver.exe'

# # wd = webdriver.Chrome(path)

# # image_url = 'https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/cute-cat-photos-1593441022.jpg?crop=0.670xw:1.00xh;0.167xw,0&resize=640:*'

# # def download_image(download_path, url, file_name):
# # 	try:
# # 		image_content = requests.get(url).content
# # 		image_file = io.BytesIO(image_content)
# # 		image = Image.open(image_file)
# # 		file_path = download_path + file_name

# # 		with open(file_path, "wb") as f:
# # 			image.save(f, "JPEG")

# # 		print("Success")
# # 	except Exception as e:
# # 		print('FAILED -', e)
# # download_image("",image_url,'cat_image.jpeg')

# # #os.mkdir("C://Users//ds_007//anaconda3//envs//web_scrapping//Images")

# # download_image("Images/", image_url, "cat_image"+ ".jpg")

# driver = webdriver.Chrome(path)
# driver.get('https://www.google.com/')

# box = driver.find_element(By.XPATH,'/html/body/div[1]/div[3]/form/div[1]/div[1]/div[1]/div/div[2]/input')
# box.send_keys('giraffe')
# box.send_keys(Keys.ENTER)

# driver.find_element(By.XPATH,'//*[@id="hdtb-msb"]/div[1]/div/div[2]/a').click()


# # Will keep scrolling down the webpage until it cannot scroll no more
# # last_height = driver.execute_script('return document.body.scrollHeight')
# # while True:
# #     driver.execute_script('window.scrollTo(0,document.body.scrollHeight)')
# #     time.sleep(2)
# #     new_height = driver.execute_script('return document.body.scrollHeight')
# #     try:
# #         driver.find_element(By.XPATH,'//*[@id="islmp"]/div/div/div/div/div[5]/input').click()
# #         time.sleep(2)
# #     except:
# #         pass
# #     if new_height == last_height:
# #         break
# #     last_height = new_height

# all_imgs = []
# keyboard = Controller()
# for i in range(1,2):
# 	try:
# 		driver.find_element(By.XPATH,'//*[@id="islrg"]/div[1]/div['+str(i)+']/a[1]/div[1]/img').click()
# 		time.sleep(10)
# 		keyboard.press(Key.enter)
# 		keyboard.release(Key.enter)
# 		for i in range(7):
# 			keyboard.press(Key.down)
# 		keyboard.press(Key.enter)
# 		keyboard.release(Key.enter)
# 		# path = "C:/Users/ds_007/anaconda3/envs/web_scrapping/Images"+"/image"+str(i)+".png"
# 		# keyboard.type(path)
# 		# keyboard.press(Key.enter)
# 		# keyboard.release(Key.enter)
# 	except Exception as e:
# 		print(e)

# driver.quit()

# # 		driver.find_element(By.XPATH, '//*[@id="Sva75c"]/div[2]/div/div[2]/div[2]/div[2]/c-wiz/div[2]/div[1]/div[1]/div[2]/div/a/img').click()
# # 		time.sleep(2)
# 		# image_content = requests.get(url).content
# 		# image_file = io.BytesIO(image_content)
# 		# image = Image.open(image_file)
# 		# cv2.imshow(image)
# 		# cv2.waitkey(0)
# 		# cv2.DestroyAllWindows()
# 		#time.sleep(5)
# 		# try:
# 		# 	image_content = requests.get(url).content
# 		# 	image_file = io.BytesIO(image_content)
# 		# 	image = Image.open(image_file)
# 		# 	file_path = "C:/Users/ds_007/anaconda3/envs/web_scrapping/Images"
# 		# 	cv2.imwrite(file_path + "/image" +str(i)+".png")
# 		# except Exception as e:
# 		# 	print("The error:",e)

# 	# with open(file_path, "wb") as f:
# 	# 	image.save(f, "JPEG")

# 	# print("Success")
# 	# except Exception as e:
# 	# 	print('FAILED -', e)
#     # print(type(image))
#     # Convert the file to an opencv image.
#     # file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
#     # cv2.imwrite('/content/drive/MyDrive/Downloading images/giraffe('+str(i)+').png',cv2.imdecode(file_bytes, 1))
#     # image.screenshot('C:/Users/ds_007/anaconda3/envs/web_scrapping/Images/giraffe('+str(i)+').png')

from selenium import webdriver
from selenium.webdriver.common.by import By
import requests
import io
from PIL import Image
import time

PATH = "C:\\Users\\Tim\\Desktop\\Web Scraping Images\\chromedriver.exe"

wd = webdriver.Chrome(PATH)

def get_images_from_google(wd, delay, max_images):
	def scroll_down(wd):
		wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
		time.sleep(delay)

	url = "https://www.google.com/search?q=tamilnadu+license+plate&tbm=isch&hl=en-GB&chips=q:tamil+nadu+number+plate,g_1:registration:st1W0aMEIko%3D,online_chips:india:uoRzdHHJsBg%3D,online_chips:high+security+number:SL-65u9txxc%3D,online_chips:plate+font:Q8MaOHqKsYg%3D&rlz=1C1CHZN_enIN970IN970&sa=X&ved=2ahUKEwiy4YnRmaj8AhWULrcAHSBTBcUQ4lYoBXoECAEQNg&biw=1519&bih=746"

	image_urls = set()
	skips = 0

	while len(image_urls) + skips < max_images:
		scroll_down(wd)

		thumbnails = wd.find_elements(By.CLASS_NAME, "Q4LuWd")

		for img in thumbnails[len(image_urls) + skips:max_images]:
			try:
				img.click()
				time.sleep(delay)
			except:
				continue

			images = wd.find_elements(By.CLASS_NAME, "n3VNCb")
			for image in images:
				if image.get_attribute('src') in image_urls:
					max_images += 1
					skips += 1
					break

				if image.get_attribute('src') and 'http' in image.get_attribute('src'):
					image_urls.add(image.get_attribute('src'))
					print(f"Found {len(image_urls)}")

	return image_urls


def download_image(download_path, url, file_name):
	try:
		image_content = requests.get(url).content
		image_file = io.BytesIO(image_content)
		image = Image.open(image_file)
		file_path = download_path + file_name

		with open(file_path, "wb") as f:
			image.save(f, "JPEG")

		#print("Success")
	except Exception as e:
		pass
		#print('FAILED -', e)

urls = get_images_from_google(wd, 1, 40)

for i, url in enumerate(urls):
	download_image("Non_Custom_Train_Images/", url, str(837+i) + ".png")

wd.quit()