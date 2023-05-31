import os
from git import (Repo, Git)
import nbformat
import glob
from dotenv import load_dotenv
from typing import (List, Dict, Optional, Tuple)
import time
import requests
from halo import Halo
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import (GPT4All, OpenAI, LlamaCpp)
from langchain.callbacks import get_openai_callback
from langchain.llms.fake import FakeListLLM
from langchain.embeddings.base import Embeddings
from langchain.llms.base import BaseLLM
import tiktoken
import torch

load_dotenv()

model_name = os.environ.get("GPT4ALL_MODEL")
llama_model_path = os.environ.get("LLAMA_MODEL_PATH")

def clone_repository(user: str, repo: str, branch: str, token: Optional[str]=None) -> str:
    """Clone the branch of a GitHub repository using the user and repo names"""

    git_url = f'https://github.com/{user}/{repo}'
    g = Git(repo)
    if token:
        git_url = git_url.replace('https://', f'https://{token}@')

    repo_folder = 'repositories/' +repo
    if not os.path.exists(repo_folder):
        os.makedirs(repo_folder)

    try:
        repo = Repo.clone_from(git_url, repo_folder, branch=branch)
    except:
        repo = Repo(repo_folder)
        actual_branch = repo.active_branch.name
        if branch != repo.active_branch.name:
            repo.git.checkout(branch)
            print(f'Checkout {actual_branch}->{branch}')
        else:
            print(f'Repository {repo} ({actual_branch}) already exists')
    return repo_folder


def get_branches(user: str, repo: str, token: Optional[str]=None) -> List[str]:
    """Get all branches of a GitHub repository"""

    headers = {'Accept': 'application/vnd.github+json'}
    if token:
        headers['Authorization'] = f'token {token}'

    response = requests.get(f'https://api.github.com/repos/{user}/{repo}/branches', headers=headers)

    branches = [branch['name'] for branch in response.json()]
    return branches


def load_notebook(file_path: str) -> Document:
    """Load a .ipynb file as a langchain Document.

    It is similar to NotebookLoader but concatenates all code cells, markdown
    cells are added as comments to the code. This produces a final document
    with less characters.
    """
    if file_path.endswith('.ipynb'):
        with open(file_path) as fp:
            notebook = nbformat.read(fp, nbformat.NO_CONVERT)
            cells = notebook['cells']
    else:
        raise ValueError('file_path must ends with .ipynb')
    full_source = []
    for cell in cells: 
        if cell['cell_type'] == 'markdown': # Add Markdown as comments
            lines = cell['source'].splitlines()
            commented_lines = ['# ' + line for line in lines]
            full_source.append('\n'.join(commented_lines))
        else:
            full_source.append(cell['source'])
    full_source = '\n\n'.join(full_source)
    doc = Document(page_content=full_source)
    doc.metadata['source'] = file_path
    parent_folder = '/'.join(file_path.split('/')[1:-1])
    if parent_folder == '':
        parent_folder = '.'
    doc.metadata['parent_folder'] = parent_folder
    return doc


def load_multiple_notebooks(source_dir: str, exclude_pattern: str='[!_]') -> List[Document]:
    """Convert all notebooks from a source_dir to langchain documents.
    It excludes all files starting with '_'.
    """
    ipynb_files = glob.glob(os.path.join(source_dir, f"{exclude_pattern}*.ipynb"), recursive=True)
    all_files = ipynb_files
    return [load_notebook(file_path) for file_path in all_files]


def generate_fake_responses(documents: List[Document]) -> Dict[str, str]:
    """Generate fake responses for FakeLLM"""
    responses = {}
    import random
    for doc in documents:
        var = 'foo' if random.randint(1,100) > 50 else 'bar'
        notebook_name = doc.metadata['source'].split('/')[-1]
        responses[notebook_name] = f'\n - `{notebook_name}`: This notebooks is used for {var}.'
    return responses


def retrieve_summary(documents: List[Document], embeddings: Embeddings, extra_context: Optional[str]='',
                      model_type: str='FakeLLM', chunk_size: int=2048, chunk_overlap: int=0, 
                      n_threads: int=4, use_gpu: bool=False, print_token_n_costs: bool=False) -> Tuple[BaseLLM,  Dict[str, Dict[str, str]]]:
    """Obtain the summary of each document using the given LLM.

    It accepts an extra_context string to include extra information of the
    document (such as meaning of acronyms, etc).
    chunk_size and chunk_overlap are used to split the document.
    print_token_n_costs is used to include the number of tokens and price per
    query when using OpenAI.
    """
    total_cost = 0
    results_parent_folder_dict={}
    chain_type="stuff"
    # Get the query for notebooks
    query = query_by_model_and_function(model_type, "notebook")
    window_context = 4001 # default davinci OpenAI

    print(f"Loading {model_type} model")
    # GPT4All models
    if model_type == 'GPT4All':
        window_context = 2048
        #callbacks = [StreamingStdOutCallbackHandler()]
        llm = GPT4All(model=model_name, temp=0.1, n_predict=256,
                      n_ctx=window_context, allow_download=True,
                      n_threads=n_threads, callbacks=None, verbose=False)
        print(f'Using {llm.backend} as backend')
        #chain_type="map_reduce"
        chunk_size = 1500
    # OpenAI model
    elif model_type == 'OpenAI':
        llm = OpenAI(temperature=0, max_tokens=500)
    # FakeLLM for testing purposes
    elif model_type == 'FakeLLM':
        responses = generate_fake_responses(documents)
        llm = FakeListLLM(responses=list(responses.values()))
    # LlaMa models from LlamaCpp, it supports GPU usage
    elif model_type == 'LlamaCpp':
        window_context = 2048
        n_gpu_layers = calculate_layer_count() if use_gpu else None
        llm = LlamaCpp(model_path=llama_model_path,  n_gpu_layers=n_gpu_layers,
                        temperature=0, max_tokens=256, n_ctx=window_context,
                        n_threads=n_threads,  callbacks=None, verbose=False)
        chunk_size = 1400
    else:
        raise ValueError('incorrect model_type (only supports OpenAI|GPT4All|LlamaCpp|FakeLLM))')
    
    # Split the documents
    for full_doc in documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                chunk_overlap=chunk_overlap,
                                                separators =  ["\n\n", "\n"])
        doc_split = text_splitter.split_documents([full_doc])
        n_tokens =[]
        for doc in doc_split:
            # remove extra break lines
            doc.page_content = doc.page_content.replace("\n\n", "\n") 
            # count number of tokens
            n_tokens.append(num_tokens_from_string(doc.page_content, "p50k_base"))

        if chain_type=="map_reduce":
            k = len(doc_split)
        else:
            k = len(doc_split) if len(doc_split)<=4 else 4
        total_tokens_worst_case = sum(sorted(n_tokens)[-k:])
        if total_tokens_worst_case > window_context:
            warning = f"WARNING: The number of tokens ({total_tokens_worst_case})"
            warning += f" to use in the embedding search ({k} documents) is"
            warning += f" likely greater that the window context ({window_context})."
            warning += " This can lead to misleading results."
            print(warning)

        # Create chromadb
        search_index = Chroma.from_documents(documents=doc_split, embedding=embeddings,)
        retriever = search_index.as_retriever(search_kwargs={"k": k})

        # Document's metadata
        notebook_name = doc.metadata['source'].split('/')[-1]
        parent_folder = doc.metadata['parent_folder']

        # Add extra context
        if extra_context:
            extra_context = f"\n{extra_context}\n"
        if parent_folder not in results_parent_folder_dict.keys():
              results_parent_folder_dict[parent_folder] = {}

        # Format the query
        query = query.format(notebook_name=notebook_name, 
                             extra_context=extra_context)
        
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type=chain_type, retriever=retriever)

        # Start the query process
        spinner = Halo(text=f'Working on {parent_folder}/{notebook_name}..', spinner='dots')
        spinner.start()
        start_time = time.time()
        cb = None
        if model_type=='OpenAI' and print_token_n_costs:
            with get_openai_callback() as cb:
                res = qa(query)
                total_cost += cb.total_cost
        elif model_type == 'FakeLLM':
            res = {'result' : responses[notebook_name]}
        elif model_type in ['GPT4All', 'LlamaCpp']:
            res = qa(query)
            res['result'] = res['result'].replace("\'", "'").replace("\\_", "_") 
            res['result'] = f"\n- `{notebook_name}`: " + res['result']
        else:
            res = qa(query)
        spinner.stop()

        final_time = round((time.time()-start_time)/60,2) # as minutes
        print(f'Running time: {final_time} min')
        if cb:
            print(cb)

        # Should not happen
        if 'ERROR: The prompt size exceeds' in res['result']:
            print(res['result'])
            exit()
        
        # Save results by parent_folder and notebook name
        results_parent_folder_dict[parent_folder][notebook_name] = res['result']

    # Print total cost, only for OpenAI API 
    if total_cost > 0:
        print(f'\nFinal cost (USD): ${total_cost}\n')

    return llm, results_parent_folder_dict


def format_summary(summary_notebooks_results: Dict[str, Dict[str, str]], repo_name: str) -> str:
    """Given the dict of results obtained from retrieve_summary, it creates a
    string formatting the results for each parent_folder.
    The output looks like this:
        **<parent_folder>** folder:
        - 00_notebook.ipynb : ...
    and so on.
    """
    summary_notebooks = '\n'
    for parent_folder in summary_notebooks_results.keys():
        llm_results = list(summary_notebooks_results[parent_folder].values())
        parent_folder = parent_folder.replace(f'{repo_name}/', '')
        if parent_folder != ".":
            summary_notebooks += f'**{parent_folder}** folder:'
        summary_notebooks += ''.join(llm_results) + '\n\n'
    return summary_notebooks


def summary_repo(llm: BaseLLM, summary_notebooks: str, repo_name: str,
                  model_type: str='FakeLLM', print_token_n_costs: bool=False) -> str:
    """Given a summary of notebooks, obtain a summary of the full repository

    print_token_n_costs is used to include the number of tokens and price of
    the query when using OpenAI.
    """
    # Get the query for the repository summary
    query_repo =  query_by_model_and_function(model_type, "repo")
    query_repo = query_repo.format(repo_name=repo_name, 
                                   summary_notebooks=summary_notebooks)
    # Start query process
    spinner = Halo(text=f'Summarizing repo..', spinner='dots')
    spinner.start()
    cb = None
    if model_type=='OpenAI' and print_token_n_costs:
        with get_openai_callback() as cb:
            summary_repo = llm(query_repo)
    elif model_type == 'FakeLLM':
            summary_repo = 'This repo contains notebooks for foo'
    else:
        summary_repo = llm(query_repo)
    spinner.stop()
    if cb:
        print(cb)

    if model_type not in ['OpenAI', 'FakeLLM']:
        summary_repo = "This repository " + summary_repo[1:]
    return summary_repo


def create_readme(repo_name: str, summary_notebooks: str, summary_repo: str) -> str:
    """Add the summary of the repo and notebooks to one string.
    The output looks like this:
    # <repo_name>
    <summary of the repository>

    Notebooks info:
    **<parent_folder>** folder:
        - 00_notebook.ipynb : ...
    and so on.
    """
    markdown_file = ""
    markdown_file += f"# {repo_name}\n"
    markdown_file += f'{summary_repo}\n\nNotebooks info:\n'
    markdown_file += f'{summary_notebooks}'
    return markdown_file



def query_by_model_and_function(model: str, to_summarize: str="notebook") -> str:
    if to_summarize == "notebook":
        if model in ['OpenAI', 'FakeLLM']:
            query = "The following code comes from the {notebook_name} Jupyter Notebook."
            query += " Your task is to explain for what this code is used for. {extra_context}"
            query += "Describe it using just one paragraph, including the location of the input and output files from the notebook if applicable.\n"
            query += "Use a bullet point for your response, which must be in markdown. "
            query += "You must don't add more information. Here is the initial part of your answer:\n"
            query += "- `{notebook_name}`: "
        elif model == 'LlamaCpp':
            query = "The following code comes from the {notebook_name} Jupyter Notebook. {extra_context}"
            query += "Do a concise summary of what this code does and include the location of the files read and written if applicable."     
        else:
            query = "Do a concise summary of what this code does. If the code read or write files, include the location of these files."
            query += "{extra_context}"
        return query
        
    elif to_summarize == "repo":
        if model in ['OpenAI', 'FakeLLM']:
            query = "The following is the summary of a list of notebooks used in a GitHub repository called `{repo_name}`:"
            query += "{summary_notebooks}"
            query += "What do you think is this repo used for? Don't explain me each notebook, just give me"
            query += " a summary of the repository that could be added to a readme.md file."""
        else:
            query = "This is a list of the summarized notebooks from the GitHub repository called `{repo_name}`:"
            query += "{summary_notebooks}"
            query += "Based on this information, a concise summary of what this repository is used for is that it "
    return query


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_gpu_memory() -> int:
    torch.cuda.empty_cache()
    """
    Returns the amount of free memory in MB for each GPU.
    """
    return int(torch.cuda.mem_get_info()[0]/(1024**2))

def calculate_layer_count() -> int:
    """
    Calculates the number of layers that can be used on the GPU.
    """
    #if not is_gpu_enabled:
    #    return None
    LAYER_SIZE_MB = 120.6 # This is the size of a single layer on VRAM, and is an approximation.
    # The current set value is for 7B models. For other models, this value should be changed.
    LAYERS_TO_REDUCE = 6 # About 700 MB is needed for the LLM to run, so we reduce the layer count by 6 to be safe.
    gpu_memory = get_gpu_memory()
    gpu_memory -= 1024 # 1GB free for security
    if (gpu_memory//LAYER_SIZE_MB) - LAYERS_TO_REDUCE > 32:
        return 32
    else:
        return (gpu_memory//LAYER_SIZE_MB-LAYERS_TO_REDUCE)
    
