import json
from bs4 import BeautifulSoup
import nltk
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')
import sys

############################
# Load Files and preprocess
############################

def remove_tags(soup):
    for data in soup(['style', 'script']):
        data.decompose()
    return ' '.join(soup.stripped_strings)

def load_topic_file(topic_filepath):
    queries = json.load(open(topic_filepath, encoding='utf-8'))
    result = {}
    for item in queries:
      title = item['Title'].lower()
      body = remove_tags(BeautifulSoup(item['Body'], "html.parser")).lower()
      result[item['Id']] = f"{title} {body}"
    return result

def load_answer_file(answer_filepath):
  lst = json.load(open(answer_filepath, encoding='utf-8'))
  result = {}
  for doc in lst:
    result[doc['Id']] = remove_tags(BeautifulSoup(doc['Text'], "html.parser")).lower()
  return result

def expand_queries(query_dict: dict, model):
    expanded_queries_dict = {}
    return expanded_queries_dict

def rewrite_queries(query_dict: dict, model):
   rewritten_queries_dict = {}
   return rewritten_queries_dict

def ranking(query_dict: dict, answer_dict: dict, run_name: str):
   pass


'''
load answers
load queries
make expanded queries
make rewritten queries
rank all and make tsv files
Evaluate in different file
'''