import argparse
import os
from dotenv import load_dotenv
from langchain.embeddings import (HuggingFaceEmbeddings, FakeEmbeddings)
import utils
load_dotenv()

embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL')
github_token = os.environ.get("GITHUB_TOKEN")
  



def main():
    args = parse_arguments()
    notebook_file = args.file
    documents = [utils.load_notebook(notebook_file)]
   
    print(f'Loading embeddings from {embeddings_model_name}..')
    if args.model == 'FakeLLM':
        embeddings = FakeEmbeddings(size=4096)
    else:
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    llm, results_pf = utils.retrieve_summary(documents, embeddings, 
                                             model_type=args.model, print_token_n_costs=True)
    print(results_pf)

def parse_arguments():
    parser = argparse.ArgumentParser(description='A description')
    parser.add_argument("--file", "-f", help='notebook path', required=True)
    parser.add_argument("--model", "-m", default='FakeLLM', help='To use a preferred model (OpenAI, FakeLLM for testing, GPT4ALL)')

    return parser.parse_args()

if __name__ == "__main__":
    if not os.path.exists('repositories/'):
        os.makedirs('repositories/')
    main()
