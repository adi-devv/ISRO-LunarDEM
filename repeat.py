from selenium import webdriver
import time

url = "https://github.com/adi-devv"
driver = webdriver.Firefox()

while True:
    driver.get(url)
