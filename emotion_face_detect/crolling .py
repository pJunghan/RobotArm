from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import urllib.request
import os

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError as e:
        print(f"Error: Failed to create the directory {directory}. {e}")

def get_image_url(driver):
    try:
        img_element = driver.find_element(By.XPATH, '//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div/div[1]/div[1]/div[2]/div[1]/a/img')
        img_url = img_element.get_attribute("src")
        if not img_url:
            img_url = img_element.get_attribute("data-src")
        if not img_url:
            img_url = img_element.get_attribute("srcset").split(',')[-1].split(' ')[0]
        print(f"Image URL: {img_url}")  # URL을 출력하여 확인
        return img_url
    except Exception as e:
        print(f"Error getting image URL: {e}")
        return None

def crawling_img(name):
    driver = webdriver.Chrome()
    driver.get("https://www.google.co.kr/imghp?hl=ko&tab=wi&authuser=0&ogbl")
    elem = driver.find_element(By.NAME, "q")
    elem.send_keys(name)
    elem.send_keys(Keys.RETURN)

    SCROLL_PAUSE_TIME = 1
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            try:
                driver.find_element(By.CSS_SELECTOR, ".mye4qd").click()
            except:
                break
        last_height = new_height

    imgs = driver.find_elements(By.CSS_SELECTOR, ".rg_i.Q4LuWd")
    dir = f"/home/hui/Downloads/image/crolling_image/{name}/"
    createDirectory(dir)
    count = 1
    for img in imgs:
        try:
            img.click()
            time.sleep(2)
            imgUrl = get_image_url(driver)
            if imgUrl:
                file_path = os.path.join(dir, f"{name}{str(count)}.jpg")
                print(f"Saving image to: {file_path}")  # 파일 경로를 출력하여 확인
                urllib.request.urlretrieve(imgUrl, file_path)
                print(f"Image saved to {file_path}")
                count += 1
                if count >= 260:
                    break
        except Exception as e:
            print(f"Error downloading image: {e}")
            continue
    driver.close()

idols = ["angry asian"]

for idol in idols:
    crawling_img(idol)
