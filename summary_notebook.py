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
    if args.model not in ["GPT4All", "OpenAI", "FakeLLM"]:
        print(f"Incorrect model {args.model}. Models supported: OpenAI|GPT4All|FakeLLM")
        exit()
    notebook_file = args.file
    documents = [utils.load_notebook(notebook_file)]
   
    print(f'Loading embeddings from {embeddings_model_name}..')
    if args.model == 'FakeLLM':
        embeddings = FakeEmbeddings(size=4096)
    else:
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    llm, results_pf = utils.retrieve_summary(documents, embeddings, n_threads=args.n_threads,
                                                model_type=args.model, print_token_n_costs=True)
    print(results_pf)

def parse_arguments():
    parser = argparse.ArgumentParser(description='A description')
    parser.add_argument("--file", "-f", help='Notebook path', required=True)
    parser.add_argument("--model", "-m", default='FakeLLM', help='To use a preferred model (OpenAI, FakeLLM for testing, GPT4All)')
    parser.add_argument("--n-threads", "-t", type=int, default=4, help='Number of threads to use, only if model==GPT4All')

    return parser.parse_args()

if __name__ == "__main__":
    if not os.path.exists('repositories/'):
        os.makedirs('repositories/')
    main()
