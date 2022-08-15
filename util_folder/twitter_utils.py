from bs4 import BeautifulSoup
import requests
import numpy as np
import pandas as pd

from selenium import webdriver
import time
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import matplotlib.pyplot as plt


class twitter_utils:
    def get_senate_names(self):
        link = 'https://www.senate.gov/senators/'
        webpage_response = requests.get(link)

        soup = BeautifulSoup(webpage_response.content, 'html.parser')

        driver = webdriver.Chrome(r'C:\Users\Memo\Desktop\For Fun Programs\TwitterBot\util_folder\chromedriver.exe')
        driver.get('https://www.senate.gov/senators/')

        x_path_lst = ['//*[@id="listOfSenators_wrapper"]/div[1]/button',
                      '//*[@id="listOfSenators_wrapper"]/div[1]/div/button[3]']

        try:
            for x_path in x_path_lst:
                element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, x_path)))
                element.click()
                # time.sleep()

                html = driver.page_source

            soup = BeautifulSoup(html, 'html.parser')
            rows = soup.find_all(attrs={'role': 'row'})

            name_state_party = []

            for row in rows[1:]:
                state_party = []
                for i, td in enumerate(row.find_all('td')):
                    state_party.append(td)
                    if i == 2:
                        break
                name_state_party.append([row.find('a').get_text(), state_party[1].get_text(), state_party[2].get_text()])

            print(name_state_party)


        except TimeoutException:
            driver.close()


if __name__ == '__main__':
    utils = twitter_utils()
    utils.get_senate_names()