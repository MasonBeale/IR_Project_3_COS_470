import csv
import json
import os
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import sys
from time import time


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

'''
############################
# Load Files and preprocess
############################
'''

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

'''
####################################
# start models and make new queries
####################################
'''

def start_model(model_id, access_token, target_directory):

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
    messages = [
        {"role": "system", "content": "You are a puzzle master, only return the keywords for each query."},
        {"role": "user", "content": "For the query: \"a riddle i found. what has one voice but goes on four legs in the morning, two in the afternoon, and three in the evening?\", Find the best keywords to represent the query."},
        {"role": "system", "content": "riddle legs day"},
        {"role": "user", "content": f"For the query \"{user_query}\" Find the best keywords to represent the query."}
    ]
    return messages
def make_messages_rewrite(user_query: str):
    messages = [
        {"role": "system", "content": "You are a puzzle master, rewrite passages you are given, give no explanation to what you changed."},
        {"role": "user", "content": "For the passage \"a riddle i found. what thing has one single voice but also goes on four legs in the morning, two legs in the afternoon, and three legs in the evening?\", rewrite the passage."},
        {"role": "system", "content": "a riddle: what has one voice but goes on four legs in the morning, two in the afternoon, and three in the evening?"},
        {"role": "user", "content": f"For the passage \"{user_query}\", rewrite the passage."}
    ]
    return messages

def expand_queries(query_dict: dict, model, tokenizer):
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


def vectorize_documents(documents, vectorizer=None):
    """Vectorize the documents (queries or answers) using TF-IDF."""
    if vectorizer is None:
        vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)
    return vectors, vectorizer


def rank_answers(answers_vectors, query_vector):
    """Rank answers based on their relevance to the query using precomputed vectors."""
    similarities = cosine_similarity(query_vector, answers_vectors).flatten()
    return similarities


# Rank answers for rewritten queries
def rank_all_queries(queries, tsv_name, run_name):
  with open(tsv_name, mode='w', newline='', encoding='utf-8') as tsv_file:
    tsv_writer = csv.DictWriter(tsv_file, fieldnames=["query_id", "Q0", "answer_id", "rank", "score", "model_name"], delimiter='\t')

    for query_id, query_text in queries.items():
        query_vector = answer_vectorizer.transform([query_text])
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


############################
# Running everything
############################

# Load data
start = time()
queries1 = load_topic_file("topics_1.json")
# queries2 = load_topic_file("topics_2.json")
answers = load_answer_file("Answers.json")
end = time()
print(f"Loading Files: {end-start} seconds")

# Precompute vectors for answers
start = time()
answer_bodies = list(answers.values())
answer_vectors, answer_vectorizer = vectorize_documents(answer_bodies)
end = time()
print(f"vectorize docs: {end-start} seconds")

# Expand and rewrite queries using LLM
start = time()
model_id = "meta-llama/Llama-3.2-3B-Instruct"  
access_token = "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"  
tokenizer, model = start_model(model_id, access_token, "./model")
end = time()
print(f"Make model: {end-start} seconds")

start = time()
rewritten_queries_1 = rewrite_queries(queries1, model, tokenizer)
end = time()
print(f"make rewritten queries: {end-start} seconds")
start = time()
expanded_queries_1 = expand_queries(queries1, model, tokenizer)
end = time()
print(f"make expanded queries: {end-start} seconds")

# expanded_queries_2 = expand_queries(queries2, model, tokenizer)
# rewritten_queries_2 = rewrite_queries(queries2, model, tokenizer)


start = time()
rank_all_queries(rewritten_queries_1,"rewritten_topic_1_results.tsv", "Rewritten Queries")
rank_all_queries(expanded_queries_1,"expanded_topic_1_results.tsv", "Expanded Queries")

# rank_all_queries(rewritten_queries_2,"rewritten_topic_2_results.tsv")
# rank_all_queries(expanded_queries_2,"expanded_topic_2_results.tsv")
end = time()
print(f"ranking: {end-start} seconds")