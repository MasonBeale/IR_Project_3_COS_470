import csv
import json
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import sys
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def remove_tags(soup):
    '''remove HTML tags'''
    for data in soup(['style', 'script']):
        data.decompose()
    return ' '.join(soup.stripped_strings)

def load_topic_file(topic_filepath):
    '''load a topic file into a dictionary'''
    queries = json.load(open(topic_filepath, encoding='utf-8'))
    result = {}
    for item in queries:
      title = item['Title'].lower()
      body = remove_tags(BeautifulSoup(item['Body'], "html.parser")).lower()
      result[item['Id']] = f"{title} {body}"
    return result

def load_answer_file(answer_filepath):
  '''load an answer file into a dictionary'''
  lst = json.load(open(answer_filepath, encoding='utf-8'))
  result = {}
  for doc in lst:
    result[doc['Id']] = remove_tags(BeautifulSoup(doc['Text'], "html.parser")).lower()
  return result

def start_model(model_id, access_token):
    '''makes a model and a tokenizer'''
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=access_token
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=access_token
    )
    model.to(device)
    return tokenizer, model

def make_messages_expand(user_query: str):
    '''creates a set of messages for a model to expand queries'''
    messages = [
        {"role": "system", "content": "You are a puzzle master, only return the keywords for each query."},
        {"role": "user", "content": "For the query: \"a riddle i found. what has one voice but goes on four legs in the morning, two in the afternoon, and three in the evening?\", Find the best keywords to represent the query."},
        {"role": "system", "content": "riddle legs day"},
        {"role": "user", "content": f"For the query \"{user_query}\" Find the best keywords to represent the query."}
    ]
    return messages

def make_messages_rewrite(user_query: str):
    '''creates a set of messages for a model to rewrite queries'''
    messages = [
        {"role": "system", "content": "You are a puzzle master, rewrite passages you are given, give no explanation to what you changed."},
        {"role": "user", "content": "For the passage \"a riddle i found. what thing has one single voice but also goes on four legs in the morning, two legs in the afternoon, and three legs in the evening?\", rewrite the passage."},
        {"role": "system", "content": "a riddle: what has one voice but goes on four legs in the morning, two in the afternoon, and three in the evening?"},
        {"role": "user", "content": f"For the passage \"{user_query}\", rewrite the passage."}
    ]
    return messages

def expand_queries(query_dict: dict, model, tokenizer):
    '''takes a dict of queries and adds some keywords for query expansion'''
    expanded_queries_dict = {}
    for query in query_dict:
        messages = make_messages_expand(query_dict[query])
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=True,
        temperature=0.1,
        top_p=0.1,
        pad_token_id=tokenizer.eos_token_id
        )
        result_text = (tokenizer.decode(outputs[0], skip_special_tokens=True)).splitlines()[-1]
        expanded_queries_dict[query] = f"{query_dict[query]} {result_text}"
    return expanded_queries_dict

def rewrite_queries(query_dict: dict, model, tokenizer):
    '''takes a dict of queries and rewrites them'''
    rewritten_queries_dict = {}
    for query in query_dict:
        messages = make_messages_rewrite(query_dict[query])
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
        )
        result_text = (tokenizer.decode(outputs[0], skip_special_tokens=True)).splitlines()[-1]
        rewritten_queries_dict[query] = f"{result_text}"
    return rewritten_queries_dict

def vectorize_documents(documents, vectorizer):
    '''takes a set of documents and turns them into a vector'''
    vectors = vectorizer.fit_transform(documents)
    return vectors

def rank_answers(answers_vectors, query_vector):
    '''takes in multiple answer vectors and one query vector and calculates the cosine simularity'''
    similarities = cosine_similarity(query_vector, answers_vectors).flatten()
    return similarities


def rank_all_queries(queries, answer_vectors, tsv_name, run_name, vectorizer):
  '''takes in queries and vectors and ranks to top 100, then adds them to a tsv'''
  with open(tsv_name, mode='w', newline='', encoding='utf-8') as tsv_file:
    tsv_writer = csv.DictWriter(tsv_file, fieldnames=["query_id", "Q0", "answer_id", "rank", "score", "model_name"], delimiter='\t')

    for query_id, query_text in queries.items():
        query_vector = vectorizer.transform([query_text])
        similarities = rank_answers(answer_vectors, query_vector)
        
        ranked_answers = sorted(zip(answers.keys(), similarities), key=lambda x: x[1], reverse=True)
        # Get the top 100 ranked answers
        for rank, (answer_id, score) in enumerate(ranked_answers[:100], start=1):
                tsv_writer.writerow({
                "query_id": query_id,
                "Q0": "Q0", 
                "answer_id": answer_id,
                "rank": rank,
                "score": score,
                "model_name": run_name
            })
    print(f"Ranked results saved to {tsv_name}")

# Load data
queries1 = load_topic_file(sys.argv[1])
queries2 = load_topic_file(sys.argv[2])
answers = load_answer_file(sys.argv[3])

# Precompute vectors for answers
answer_texts = list(answers.values())
vectorizer = TfidfVectorizer()
a_vecs = vectorize_documents(answer_texts, vectorizer)

# Expand and rewrite queries using LLM
model_id = "meta-llama/Llama-3.2-3B-Instruct"  
access_token = "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"  
tokenizer, model = start_model(model_id, access_token)

# Making new queries
rewritten_queries_1 = rewrite_queries(queries1, model, tokenizer)
rewritten_queries_2 = rewrite_queries(queries2, model, tokenizer)
expanded_queries_1 = expand_queries(queries1, model, tokenizer)
expanded_queries_2 = expand_queries(queries2, model, tokenizer)

# Ranking all queries
rank_all_queries(rewritten_queries_1, a_vecs, "rewritten_topic_1.tsv", "Rewritten Queries", vectorizer)
rank_all_queries(expanded_queries_1, a_vecs, "expanded_topic_1.tsv", "Expanded Queries", vectorizer)
rank_all_queries(rewritten_queries_2, a_vecs, "rewritten_topic_2.tsv", "Rewritten Queries", vectorizer)
rank_all_queries(expanded_queries_2, a_vecs, "expanded_topic_2.tsv", "Expanded Queries", vectorizer)