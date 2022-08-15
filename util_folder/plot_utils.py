from bs4 import BeautifulSoup
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from selenium import webdriver
import time
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import matplotlib.pyplot as plt

import opendatasets as od
import collections
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

import plot_utils
import statistics

from nltk.sentiment.vader import SentimentIntensityAnalyzer




def plot_topics(topic1, topic2, topic_names):
    """
    Creates a bar graph between two topics. Topic names contains the names of the list
    ARGS:
        Topic1 (list): topic one (list of relevant terms)
        Topic2 (list): topic two (list of sentiment of each relevant term)
        Topic_names (list): Contains the x and y axis labels
    Returns:
        A Bar Graph between topic1 vs topic2.
    
    """
    fig, ax = plt.subplots(figsize=(16,9))
    colors = []
    
    for val in topic2:
        colors.append('red' if val < 0 else 'purple')
    
    
    ax.barh(topic1, topic2, color=colors)

    # remove x, y ticks
    ax.xaxis.set_ticks_position('none'), ax.yaxis.set_ticks_position('none')

    # Add x, y, gridlines
    ax.grid(b=True, color='grey', linestyle='-', linewidth=0.5, alpha=0.2)

    ax.invert_yaxis()

    # Add annotation to bars
    for i in ax.patches:
        if i.get_width() < 0:
            plt.text(i.get_width()-20, i.get_y()+0.5,
                 str(round((i.get_width()), 2)),
                 fontsize = 10, fontweight ='bold',
                 color ='red')
        else:
            plt.text(i.get_width()+0.2, i.get_y()+0.5,
                 str(round((i.get_width()), 2)),
                 fontsize = 10, fontweight ='bold',
                 color ='grey')
        

    ax.set_title(f'{topic_names[0]} vs {topic_names[1]}')
    ax.set_xlabel(f'{topic_names[1]}'),ax.set_ylabel(f'{topic_names[0]}')

    plt.show()
 
def plot_LDA_topics(count, LDA_component, n_components=1):
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=False)
    axes = axes.flatten()
    
    for index, topic in enumerate(LDA.component):
        top_features_weight = topic.argsort()[-10:]
        weights = topic[top_features_weight]
        names = [count.get_feature_names()[i] for i in topic.argsort()[-10:]]
              
        ax = axes[index]
        ax.barh(names, weights, color='purple')
        ax.set_title(f'Topic {index+1}'), ax.set_xlabel('Weights'), ax.set_ylabel('Word')
                 
        for i in ax.patches:
                 plt.text(i.get_width()+0.2, i.get_y()+0.5,
                 str(round((i.get_width()), 2)),
                 fontsize = 10, fontweight ='bold',
                 color ='grey')
    plt.tight_layout()
    plt.show()
        


def plot_sent_frequency(sent, freq, names):
    fig, ax = plt.subplots(figsize=(20, 9))
    sns.scatterplot(sent, freq)

    ax.grid(b=True, color='grey', linestyle='-', linewidth=0.5, alpha=0.2)
    ax.set_facecolor('lightgrey')

    texts = []
    for label, x,  y, in zip(names, sent, freq):
        """
        args:

        Label: contains the text we want to annotate the point with
        x: the x-value/coord of the point
        y: the y-value/coord of the point

        Returns:
            Annotation of each point
        """
        texts.append(plt.text(x, y, label))

    ax.set_xlabel('Sentiment'), ax.set_ylabel('Frequency')    
    ax.set_title('Sentiment vs Frequency')

    adjust_text(texts, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
    
    plt.show()
   

def get_senate_names():
    """
    Scrape the name of current senators from www.senate.gov
    """
    
    link = 'https://www.senate.gov/senators/'
    webpage_response = requests.get(link)

    soup = BeautifulSoup(webpage_response.content, 'html.parser')

    driver = webdriver.Chrome(r'chromedriver.exe')
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

        return name_state_party


    except TimeoutException:
        driver.close()
        
        
def get_compound_score(impact_words, df):
    """
    Given a list of terms, find the compound score of each tweet that contains the term
    
    Returns:
        dictionary that contains term vs sentiment
    """
    
    
    words = ' '.join(df['text']).lower()
    counter = collections.Counter(words.split())
    data = {}
    
    
    for word in impact_words:
        data[word] = [counter[word], sum(df.loc[df['text'].str.contains(word)]['Scores'])]
        
    return data 

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

