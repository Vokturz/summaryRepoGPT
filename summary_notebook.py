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
    if args.model not in ["GPT4All", "OpenAI", "FakeLLM", "LlamaCpp"]:
        print(f"Incorrect model {args.model}. Models supported: OpenAI|GPT4All|FakeLLM|LlamaCpp")
        exit()
    notebook_file = args.file
    documents = [utils.load_notebook(notebook_file)]
   
    print(f'Loading embeddings from {embeddings_model_name}..')
    if args.model == 'FakeLLM':
        embeddings = FakeEmbeddings(size=4096)
    else:
        if args.gpu and args.model == 'LlamaCpp':
            embeddings_kwargs = {'device': 'cuda'}
        else:
            embeddings_kwargs = {}
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name,
                                            model_kwargs=embeddings_kwargs)
    llm, results_pf = utils.retrieve_summary(documents, embeddings, n_threads=args.n_threads,
                                             use_gpu=args.gpu, model_type=args.model,
                                             print_token_n_costs=True)
    print(results_pf)

def parse_arguments():
    parser = argparse.ArgumentParser(description='A description')
    parser.add_argument("--file", "-f", help='Notebook path', required=True)
    parser.add_argument("--model", "-m", default='FakeLLM',
                         help='To use a preferred model (OpenAI, FakeLLM for testing, GPT4All, LlamaCpp)')
    parser.add_argument("--n-threads", "-t", type=int, default=4,
                         help='Number of threads to use')
    parser.add_argument("--gpu", "-g", type=bool, default=False,
                        help='To run using GPU (Only for LlamaCpp)')
    return parser.parse_args()

if __name__ == "__main__":
    if not os.path.exists('repositories/'):
        os.makedirs('repositories/')
    main()
