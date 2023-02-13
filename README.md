# Web-scraping
twitter web scraping in python

!pip install bs4

#Task 1
from bs4 import BeautifulSoup as bs
import requests 
from csv import writer
import pandas as pd
import matplotlib.pyplot as plt 

url='https://www.amazon.in/s?k=python+books&crid=CPQ27XOAZ9VS&sprefix=python+books%2Caps%2C809&ref=nb_sb_noss_1'
headers={'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36'}

page=requests.get(url, headers=headers)
soup=bs(page.content,'html.parser')
#get all books
books=soup.find_all('div', class_="s-result-item s-asin sg-col-0-of-12 sg-col-16-of-20 sg-col s-widget-spacing-small sg-col-12-of-16")

with open('books.csv','w', encoding='utf8', newline='') as f:
  thewriter=writer(f)
  header=['name','author','price','ratings']
  thewriter.writerow(header)

  for book in books:
    name=book.find('span', class_="a-size-medium a-color-base a-text-normal").text
    price=book.find('span', class_="a-price-whole").text
    rating=book.find('span', class_="a-icon-alt").text[:3]
    author=book.find('div',class_="a-row").text[3:]
    info=[name,author,price,rating]
    thewriter.writerow(info)

df=pd.read_csv('/content/books.csv')
df['price'] = df['price'].str.replace(',','')
df['price'] = df['price'].str.replace('.','').astype(int)
price=df.plot.bar(y='price',rot=0)
plt.xlabel('Book name')
plt.ylabel('ratings')
ratings=df.plot.bar(y='ratings',rot=0)
plt.xlabel('Book name')
plt.ylabel('Rating')
display(df)

#text process
import nltk                             
from nltk.corpus import twitter_samples   
import matplotlib.pyplot as plt           
import random  
import pandas as pd
df

#Removing punctuation
import string
string.punctuation

#Defining Function to remove punctuation
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree
#storing the puntuation free text
df['clean_tweets']= df['name'].apply(lambda x:remove_punctuation(x))
df.head()

#Lowering the tweets
df['lower_tweets']= df['clean_tweets'].apply(lambda x: x.lower())
df.head()

#Tokenization
from nltk.tokenize import TweetTokenizer as tt
#applying function to the column
tokenizer = tt()      # instantiate the tokenizer class
df['tokenized_tweets']=df['lower_tweets'].apply(lambda x: tokenizer.tokenize(x))

#Removing stop words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopword = stopwords.words('english')

#Defining the function to remove stopwords from tokenized text
def remove_stopwords(text):
    output= [i for i in text if i not in stopword]
    return output
#applying the function
df['no_stopwords']= df['tokenized_tweets'].apply(lambda x:remove_stopwords(x))

#Stemming 
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
#Defining a function for stemming
def stemming(text):
    stem_tweet = [porter_stemmer.stem(word) for word in text]
    return stem_tweet
df['stemmed_tweets']=df['no_stopwords'].apply(lambda x: stemming(x))

#Lemmatization
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
wordnet_lemmatizer = WordNetLemmatizer()

#Defining the function for lemmatization
def lemmatizer(text):
    lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
    return lemm_text
df['lemmatized_tweets']=df['stemmed_tweets'].apply(lambda x:lemmatizer(x))
df.head()
df.to_csv("/content/drive/MyDrive/train/house.csv")
import matplotlib.pyplot as plt
import collections
from wordcloud import WordCloud

listed=df.lemmatized_tweets.sum()
corrected=[k for k, v in collections.Counter(listed).items() if v > 2]
print(corrected)
unique_string=(" ").join(corrected)
wordcloud = WordCloud(width = 1000, height = 700).generate(unique_string)
plt.figure(figsize=(15,7))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
plt.close()
