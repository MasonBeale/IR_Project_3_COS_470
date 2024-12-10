# IR_Project_3_COS_470
## Project Overview

This project uses a combination of natural language processing (NLP) and machine learning techniques to rank answers for queries. It includes functionalities for:

- Loading and preprocessing topic and answer files.
- Expanding or rewriting queries using a language model.
- Vectorizing documents with TF-IDF.
- Ranking answers based on cosine similarity.
- Saving ranked results to a TSV file.

## Contributions
Mason: Worked on testing different prompt inputs/messages for the model.
Abbas: Made the TF-IDF vectorization results methods. 

Code was run on Mason's computer, Abbas wasn't able to run the code on the USM lab computers.

## Prerequisites

### Required Libraries
The following libraries are required to run this project:
- `transformers`
- `torch`
- `scikit-learn`
- `tqdm`
- `beautifulsoup4`
- `csv`
- `json`
- `pickle`
- `os`
- `sys`

Install missing libraries with `pip`:
```bash
pip install transformers torch scikit-learn tqdm beautifulsoup4
```

### Hardware Requirements
A machine with GPU support is recommended for running large language models efficiently. The project automatically detects CUDA availability.

## File Structure
- `topics_1.json`: Topic file containing queries with fields like `Id`, `Title`, and `Body`.
- `Answers.json`: Answer file containing answer data with fields like `Id` and `Text`.
- `rewritten_topic_1_results.tsv`: Output file for ranked answers based on rewritten queries.

## Features

### Preprocessing
- **HTML Tag Removal:** Strips HTML tags using BeautifulSoup to clean the `Body` and `Text` fields from JSON files.
- **Case Normalization:** Converts text to lowercase.

### Query Expansion and Rewriting
- **Query Expansion:** Generates expanded queries using a language model.
- **Query Rewriting:** Rewrites queries for better clarity and relevance.

### Document Vectorization
Uses **TF-IDF Vectorization** to convert text into numerical vectors for similarity calculation.

### Ranking Answers
Calculates cosine similarity between query and answer vectors to rank answers. Saves the top 100 ranked answers for each query in a TSV file.

## Usage

1. **Load the Data**
   - Ensure `topics_1.json` and `Answers.json` are in the same directory as the script.
   - The data is preprocessed during the file loading stage.

2. **Start the Model**
   - Set your model details (`model_id` and `access_token`) in the script.
   - The model and tokenizer will be downloaded and saved locally to the `./model` directory.

3. **Run Query Expansion or Rewriting**
   - The script uses predefined functions to expand or rewrite queries using the loaded model.

4. **Rank Answers**
   - Precomputed TF-IDF vectors of answers are compared with query vectors to rank answers.
   - Results are saved in a TSV file.

### Example Commands
Run the script:
```bash
python Project3.py topics_1.json topics_2.json Answers.json
```

Expected Outputs:
- Preprocessing times for topics and answers.
- Time taken for model loading and query rewriting.
- Saved ranking results in a file like `rewritten_topic_1_results.tsv`.

## Output Format
The TSV file contains the following fields:
| Field       | Description                           |
|-------------|---------------------------------------|
| `query_id`  | The ID of the query                   |
| `Q0`        | Placeholder for compatibility         |
| `answer_id` | The ID of the ranked answer           |
| `rank`      | Rank of the answer                    |
| `score`     | Cosine similarity score               |
| `model_name`| Name of the model used for ranking    |

## Customization
### Modify Language Model
You can change the model by updating the `model_id` and `access_token` in the script. Ensure the model supports `AutoModelForCausalLM`.

### Adjust Ranking
To adjust the number of answers saved:
- Edit the `rank_all_queries` function, modifying `ranked_answers[:100]` to a desired number.

## Author
Abbas Jabor and Mason Beale
