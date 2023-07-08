import re
import scrapy
import pandas as pd
from scrapy.crawler import CrawlerProcess
import chompjs
import random


user_agent_list = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.82 Safari/537.36',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_4_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Mobile/15E148 Safari/604.1',
    'Mozilla/4.0 (compatible; MSIE 9.0; Windows NT 6.1)',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36 Edg/87.0.664.75',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.18363',
]


class Spider20Minutes(scrapy.Spider):
    name = '20minutes'
    start_urls = ['https://www.20minutes.fr/archives/20' + str(i) for i in range(10, 23)] #+ ['https://www.20minutes.fr/archives/200' + str(i) for i in range(6, 10)]
    
    custom_settings = {
        'FEED_URI': '../data/newspaper_2.jsonl',
        'FEED_FORMAT': 'jsonl',
        'FEED_EXPORT_ENCODING': 'utf-8'
    }

    def parse(self, response):
        links = response.xpath('//div[@class="box brick mb2 month"]//@href').getall()
        for link in links:
            yield response.follow(link, callback = self.parse_day)

    def parse_day(self, response):
        links = response.xpath('//ul[@class="spreadlist"]//@href').getall()
        for link in links:
            yield response.follow(link, callback = self.parse_link, cb_kwargs={'url': 'https://www.20minutes.fr' + link})

    def parse_link(self, response, **kwargs):
        link = kwargs['url']
        title = response.xpath('//article[@id="main-content"]//h1[@class="nodeheader-title"]/text()').get()
        summary = response.xpath('//article[@id="main-content"]//span[@class="hat-summary"]/text()').get()
        article_date = response.xpath('//article[@id="main-content"]//div[@class="datetime"]//@datetime').get()
        body_html = response.xpath('//article[@id="main-content"]//div[@class="lt-endor-body content"]//div[@class="qiota_reserve content"]/p//text() | //article[@id="main-content"]//div[@class="lt-endor-body content"]//div[@class="qiota_reserve content"]/h2//text() | //article[@id="main-content"]//div[@class="lt-endor-body content mt1"]//p//text()').getall()

        body = " ".join(body_html).replace("\xa0", " ")
        
        image_url = response.xpath('//article[@id="main-content"]//div[@class="lt-endor-body content"]//div[@class="media-wrap"]//@src').get()
        
        try:
            category = link.split('.fr/')[1].split('/')[0]

        except:
            category= "undefined"
        date = pd.to_datetime('today')


        yield {
            'article_url': link,
            'title': title,
            'summary': summary,
            'article_date': article_date,
            'body': body,
            'image_url': image_url,
            'category_id': category,
            'journal_id': 1,
            'scraping_date': str(date)
        }
        

        
        
class SpiderLiberation(scrapy.Spider):
    name = 'Liberation'
    start_urls = ['https://www.liberation.fr/archives/20' + str(i) +'/' for i in range(10, 23)] + ['https://www.liberation.fr/archives/200' + str(i) +'/' for i in range(6, 10)]
    #start_urls = ['https://www.liberation.fr/archives/2022']
    custom_settings = {
        'FEED_URI': 'data/newspaper_liberation.jsonl',
        'FEED_FORMAT': 'jsonl',
        'FEED_EXPORT_ENCODING': 'utf-8'
    }

    def parse(self, response):
        links = response.xpath('//div[@class="flex flex_col"]//@href').getall()
        for link in links:
            yield response.follow(link, callback = self.parse_month)

    def parse_month(self, response):
        links = response.xpath('//div[@class="flex flex_col"]//@href').getall()
        for link in links:
            yield response.follow(link, callback = self.parse_day)

    
    def parse_day(self, response):
        links = response.xpath('//div[@class="custom-card-list"]//@href').getall()
        for link in links:
            yield response.follow(link, callback = self.parse_link, cb_kwargs={'url': 'https://www.liberation.fr' + link})

    def parse_link(self, response, **kwargs):
        link = kwargs['url']
        js_object_dico = chompjs.parse_js_object(response.xpath('//article[contains(@class,"article-body-wrapper")]/script[1]/text()').get())
        try:
            title = js_object_dico["headline"]
        except:
            yield({'Error' : 'No Headline'})
        #title = response.xpath('//h1[contains(@class,"TypologyArticle__BlockTitleHeadline")]/text()').get()
        summary = js_object_dico["description"]
        #summary = response.xpath('//span[contains(@class,"TypologyArticle__BlockSubHeadline")]/text()').get()
        article_date = js_object_dico["datePublished"]
        body_html = response.xpath('//article[contains(@class,"article-body-wrapper")]//p/text() | //article[contains(@class,"article-body-wrapper")]//h2/text()').getall()

        body = " ".join(body_html).replace("\xa0", " ")
        
        try :
            image_url = js_object_dico["image"]["url"]
        except:
            image_url = "None"

        try:
            category = link.split('.fr/')[1].split('/')[0]

        except:
            category= "undefined"
        date = pd.to_datetime('today')
        

        yield {
            'article_url': link,
            'title': title,
            'summary': summary,
            'article_date': article_date,
            'body': body,
            'image_url': image_url,
            'category_id': category,
            'journal_id': 2,
            'scraping_date': str(date)
        }
    
    

      
class SpiderLeMonde(scrapy.Spider):
    name = 'LeMonde'
    start_urls = ['https://www.lemonde.fr/archives-du-monde/' + str(d) +'-'+ str(m) +'-'+ str(y) +'/' for d in range(1, 32) for m in range(1, 13) for y in range(2010, 2023)]
    #start_urls = ['https://www.lemonde.fr/archives-du-monde/12-09-2001/']
    custom_settings = {
        'FEED_URI': 'data/newspaper_lemonde.jsonl',
        'FEED_FORMAT': 'jsonl',
        'FEED_EXPORT_ENCODING': 'utf-8',
        }
    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url=url, callback=self.parse,
                        headers={"User-Agent": user_agent_list[random.randint(0, len(user_agent_list)-1)]})
    
    def parse(self, response):
        links = response.xpath('//section[@class="river__pagination"]//@href').getall()
        for link in links:
            yield response.follow("https://www.lemonde.fr/" + link, callback = self.parse_pagination,)

    def parse_pagination(self, response):
        links = response.xpath('//section[@class="teaser teaser--inline-picture  "]//@href').getall()
        for link in links:
            yield response.follow(link, callback = self.parse_link, cb_kwargs={'url': link})
            
    def parse_link(self, response,**kwargs):
        link = kwargs['url']
        title = response.xpath('//h1[@class="article__title"]/text()').get()
        summary = response.xpath('//p[@class="article__desc"]/text()').get()
        article_date = response.xpath('//meta[@property="og:article:published_time"]//@content').get()
        body_html = response.xpath('//article[@class="article__content old__article-content-single"]//h2//text() | //article[@class="article__content old__article-content-single"]//p//text()').getall()

        body = " ".join(body_html).replace("\xa0", " ")

        image_url = response.xpath('//meta[@property="og:image"]//@content').get()

        try:
            category = link.split('.fr/')[1].split('/')[0]

        except:
            category= "undefined"
        date = pd.to_datetime('today')
        

        yield {
            'article_url': link,
            'title': title,
            'summary': summary,
            'article_date': article_date,
            'body': body,
            'image_url': image_url,
            'category_id': category,
            'journal_id': 3,
            'scraping_date': str(date)
        }
    
    


    
if __name__=='__main__':
    process = CrawlerProcess()
    process.crawl(Spider20Minutes)
    process.start()

if __name__=='__main__':
    process = CrawlerProcess()
    process.crawl(SpiderLiberation)
    process.start()

if __name__=='__main__':
    process = CrawlerProcess()
    process.crawl(SpiderLeMonde)
    process.start()
