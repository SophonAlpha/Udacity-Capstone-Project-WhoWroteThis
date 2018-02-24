from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import WebDriverException
import time
import json
import os
from pathlib import Path
import re

AUTHORS_FILE = 'Medium_authors_25.txt'
ARTICLE_URLS_FILE = 'Medium_article_urls.json'
ARTICLES_FOLDER = 'Medium_articles'

class MediumArticleCollector:
    
    def __init__(self, browserType='Firefox'):
        self.wait_timeout = 300
        self.scroll_down_wait = 1
        self.browserType = browserType
        self.setup_browser()
        self.author_URLs = []
        self.article_URLs = self.load_saved_article_URLs()
        self.articles = []

    # --------------- methods to collect article URLs --------------------------

    def save_article_URLs(self):
        for author_URL in self.get_next_author_URL():
            if not(author_URL):
                author_URL, article_URLs = self.get_author_article_URLs(author_URL)
                self.write_article_URLs_to_file(author_URL, article_URLs)

    def setup_browser(self):
        if self.browserType == 'Firefox':
            self.browser = webdriver.Firefox(
                 executable_path='C:\Program Files\Geckodriver\geckodriver.exe')
        if self.browserType == 'Chrome':
            self.browser = webdriver.Chrome(
                executable_path='C:\Program Files (x86)\Google\Chromedriver\chromedriver.exe')
        self.browser.get('http://www.medium.com/')
        # Medium.com login is not automated. The login needs to be performed by 
        # the user. Medium offers multiple options for login. The user needs to 
        # make sure he logs in the same tab that was opened by this script.
        # Not ideal. Maybe I build an automated solution later.
        print('Please login to medium.com.')
        self.wait_for_element('//img[@class="avatar-image avatar-image--icon"]')

    def wait_for_element(self, element_xpath):
        element = WebDriverWait(self.browser, self.wait_timeout).until(
            EC.presence_of_element_located((By.XPATH, element_xpath)))
        return element

    def get_next_author_URL(self):
        f = open(AUTHORS_FILE, 'r')
        while True:
            author_URL = f.readline()
            yield author_URL
            if not(author_URL):
                break
        f.close()

    def load_saved_article_URLs(self):
        if Path(ARTICLE_URLS_FILE).exists():
            with open(ARTICLE_URLS_FILE, 'r') as f:
                article_URLs = json.load(f)
        else:
            article_URLs = []
        return article_URLs
    
    def get_next_article_URL(self):
        for article_URL in self.article_URLs:
            yield article_URL

    def remove_authors_already_covered(self, author_urls):
        reduced_list = []
        author_urls_already_covered = \
            list(set([article_url['author_URL'] for article_url in self.article_URLs]))
        for author_url in author_urls:
            if author_url in author_urls_already_covered:
                pass
            else:
                reduced_list.append(author_url)
        return reduced_list

    def write_article_URLs_to_file(self, author_URL, article_URLs):
        for article_URL in article_URLs:
            if not(self.article_URL_in_list(article_URL)):
                self.article_URLs.append({'author_URL': author_URL,
                                          'article_URL': article_URL})
            with open(ARTICLE_URLS_FILE, 'w') as f:
                json.dump(self.article_URLs, f)
        with open('saved_Medium_authors.csv', 'a+') as f:
            f.write('{};{}\n'.format(author_URL.rstrip(), len(article_URLs)))
        print('{} = {} articles'.format(author_URL.rstrip(), len(article_URLs)))

    def article_URL_in_list(self, article_URL):
        found = False
        for a in self.article_URLs:
            if a['article_URL'] == article_URL:
                found = True
                break
        return found

    def get_author_article_URLs(self, author_URL):
        '''
        Navigate to the authors list of articles.
        Return list of all URLs to articles for the author.
        '''
        try:
            self.browser.get(author_URL)
        except WebDriverException as err:
            if err.msg.find('Reached error page:') >= 0:
                # Occasionally elements of the Medium page are not loaded and articles_page
                # "server side reset" exception is thrown by the WebDriver.
                # This exception handler catches and ignores this.
                print('encountered WebDriverException trying to continue. {0}'.format(err.msg))
            else:
                raise err
        self.build_article_links_page()
        article_URLs = self.extract_article_URLs()
        return author_URL, article_URLs

    def build_article_links_page(self):
        '''
        Build complete list of articles by scrolling down until list is complete.
        '''
        while True:
            height_before_scroll = self.browser.execute_script('return document.body.scrollHeight')
            self.browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            if self.reached_end_of_page(height_before_scroll):
                break
        return
    
    def reached_end_of_page(self, height_before_scroll):
        '''
        Do an escalating wait to ensure page has been updated after the scroll
        down command and we really have reached the end of the page. Wait time
        is escalated to avoid unnecessary long wait time with faster page loads.
        '''
        max_wait_cycles = 5
        reached_end_of_page = False
        for factor in range(1, max_wait_cycles+1):
            height_after_scroll = self.browser.execute_script('return document.body.scrollHeight')
            if height_after_scroll == height_before_scroll:
                reached_end_of_page = True
                wait_time = self.scroll_down_wait*factor
                time.sleep(wait_time)
                self.browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            else:
                reached_end_of_page = False
                break
        return reached_end_of_page
    
    def extract_article_URLs(self):
        '''
        Extract the article URLs.
        '''
        article_URLs = self.browser.find_elements_by_xpath("//a[@class='button button--smaller button--chromeless u-baseColor--buttonNormal']")
        article_URLs = [article.get_attribute('href') for article in article_URLs]
        return article_URLs

    # --------------- methods to download the articles -------------------------

    def save_articles(self):
        '''
        Download all articles.Return list of articles.
        '''
        article_URLs = self.load_saved_article_URLs()
        article_URLs = self.remove_articles_already_saved(article_URLs)
        for entry in article_URLs:
            article = self.get_article(entry['article_URL'])
            if not(article == None):
                self.save_article_to_file(article)
                print('{} | {} | {} bytes | {}'.format(article['author'],
                                                article['headline'],
                                                len(article['body']),
                                                article['url']))

    def remove_articles_already_saved(self, article_URLs):
        saved_article_URLs = self.get_saved_article_URLs()
        reduced_list = self.remove_done_URLs(article_URLs, saved_article_URLs)
        return reduced_list

    def get_saved_article_URLs(self):
        saved_article_URLs = []
        for root, _, files in os.walk(ARTICLES_FOLDER):
            for file in files:
                article_file = os.path.join(root, file)
                with open(article_file, 'r') as f:
                    article = json.load(f)
                    saved_article_URLs.append(article['url'])
        return saved_article_URLs

    def remove_done_URLs(self, article_URLs, saved_article_URLs):
        reduced_list = []
        for article_URL in article_URLs:
            if not(article_URL['article_URL'] in saved_article_URLs):
                reduced_list.append(article_URL)
        return reduced_list

    def get_article(self, article_url):
        '''
        Download one article.
        '''
        article = None
        try:
            self.browser.get(article_url)
        except WebDriverException as err:
            if err.msg.find('Reached error page:') >= 0:
                print('encountered WebDriverException trying to continue. {0}'.format(err.msg))
            else:
                raise err
        else:
            try:
                author = self.browser.find_element_by_xpath('//a[@rel="author cc:attributionUrl"]').text
                headline = self.get_headline()
                body = self.browser.find_element_by_xpath('//div[@class="postArticle-content js-postField js-notesSource js-trackedPost"]').text
                article = {'url': article_url,
                           'author': author,
                           'headline': headline,
                           'body': body}
            except WebDriverException as err:
                print('encountered WebDriverException trying to continue. {0}'.format(err.msg))
        return article
    
    def get_headline(self):
        headline =''
        for i in range(1,5):
            headline = self.browser.find_elements_by_tag_name('h{}'.format(i))
            if len(headline) > 0:
                headline = headline[0].text
                break
        return headline

    def save_article_to_file(self, article):
        folders = self.to_file_string(os.path.join(ARTICLES_FOLDER, article['author']))
        m = re.search('.*/(?P<file>.*)\?', article['url'])
        file = os.path.join(folders, m.group('file')+'.json')
#        file = self.to_file_string(os.path.join(folders, unidecode(article['headline'])+'.json'))
        if not(os.path.exists(folders)):
            os.makedirs(folders)
        with open(file, 'w') as f:
            json.dump(article, f)
            
    def to_file_string(self, string):
        non_file_chars = '''[~#%&*{}\\:<>?/+|\"']'''
        string = re.sub(non_file_chars, '', string)
        string = re.sub('[\s]', '_', string)
        return string

    def close(self):
        self.browser.quit()

# ------------------- Main -----------------------------------------------------

if __name__ == "__main__":
    print('Start')
    Medium = MediumArticleCollector()
    if False:
        # Run this part to build the list of article URLs.
        startTime = time.time()
        Medium.save_article_URLs()
        elapsedTime = time.time() - startTime
        print('Article URLs collected in {0} hours.'.format(elapsedTime/3600))
    if True:
        # Run this part when downloading the articles. This requires the list
        # of article URLs to be build beforehand.
        startTime = time.time()
        Medium.save_articles()
        elapsedTime = time.time() - startTime
        print('Articles downloaded in {0} hours.'.format(elapsedTime/3600))
    Medium.close()
