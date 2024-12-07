import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from bs4 import BeautifulSoup
import sys

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
def start_model(model_id, access_token):
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
    return model, tokenizer

def make_messages_expand(user_query: str):
    messages = [
        {"role": "system", "content": "You are a puzzle master, only return the keywords for each query"},
        {"role": "user", "content": "For the query: \"a riddle i found. what has one voice but goes on four legs in the morning, two in the afternoon, and three in the evening?\", Find the best keywords to represent the query."},
        {"role": "system", "content": "riddle legs day"},
        {"role": "user", "content": f"For the query \"{user_query}\" Find the best keywords to represent the query."}
    ]
    return messages
def make_messages_rewrite(user_query: str):
    messages = [
        {"role": "system", "content": "You are a puzzle master"},
        {"role": "user", "content": "For the query \"a riddle i found. what thing has one single voice but also goes on four legs in the morning, two legs in the afternoon, and three legs in the evening?\", rewrite the query to be more clear and concise."},
        {"role": "system", "content": "a riddle: what has one voice but goes on four legs in the morning, two in the afternoon, and three in the evening?"},
        {"role": "user", "content": f"For the query\"{user_query}\", rewrite the query to be more clear and concise."}
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
        max_new_tokens=200,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
        )
        result_text = (tokenizer.decode(outputs[0], skip_special_tokens=True)).splitlines()[-1]
        rewritten_queries_dict[query] = f"{result_text}"
   return rewritten_queries_dict

'''
#############################
Ranking and making tsv files
#############################
'''

def ranking(query_dict: dict, answer_dict: dict, run_name: str):
   pass


'''
###################
Running everything
###################
load answers
load queries
make expanded queries
make rewritten queries
rank all and make tsv files
Evaluate in different file
'''
model, tokenizer = start_model("meta-llama/Llama-3.2-3B-Instruct", "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")

topics = load_topic_file("topics_1.json")
one_topic = {"49160": topics["49160"]}

print(expand_queries(one_topic, model, tokenizer))