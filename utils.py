import os
from git import (Repo, Git)
import nbformat
import glob
from dotenv import load_dotenv
from typing import (List, Dict, Optional, Tuple)
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import (GPT4All, OpenAI)
from langchain.callbacks import get_openai_callback
from langchain.llms.fake import FakeListLLM
from langchain.embeddings.base import Embeddings
from langchain.llms.base import BaseLLM
load_dotenv()

model_name = os.environ.get("GPT4ALL_MODEL")


def load_notebook(file_path: str) -> Document:
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
    full_source = '\n'.join(full_source)
    doc = Document(page_content=full_source)
    doc.metadata['source'] = file_path
    doc.metadata['parent_folder'] = '/'.join(file_path.split('/')[1:-1])
    return doc


def load_multiple_notebooks(source_dir: str, exclude_pattern: str='[!_]') -> List[Document]:
    ipynb_files = glob.glob(os.path.join(source_dir, f"{exclude_pattern}*.ipynb"), recursive=True)
    all_files = ipynb_files
    return [load_notebook(file_path) for file_path in all_files]

def create_readme(repo_name: str, summary_notebooks: str, summary_repo: str) -> str:
    markdown_file = ""
    markdown_file += f"# {repo_name}\n"
    markdown_file += f'{summary_repo}\n\nNotebooks info:\n'
    markdown_file += f'{summary_notebooks}'
    return markdown_file


def format_summary(summary_notebooks_results: Dict[str, Dict[str, str]], repo_name: str) -> str:
    summary_notebooks = '\n'
    for parent_folder in summary_notebooks_results.keys():
        llm_results = list(summary_notebooks_results[parent_folder].values())
        parent_folder = parent_folder.replace(f'{repo_name}/', '')
        summary_notebooks += f'**{parent_folder}** folder:'
        summary_notebooks += ''.join(llm_results) + '\n\n'
    return summary_notebooks


def clone_repository(repo_username: str, repo_name: str,
                      branch: str, token: Optional[str]=None) -> str:
    git_url = f'https://github.com/{repo_username}/{repo_name}'
    g = Git(repo_name)
    if token:
        git_url = git_url.replace('https://', f'https://{token}@')

    repo_folder = 'repositories/' +repo_name
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
            print(f'Repository {repo_name} ({actual_branch}) already exists')
    return repo_folder

def get_branches(user: str, repo: str, token: Optional[str]=None) -> List[str]:
    headers = {'Accept': 'application/vnd.github+json'}
    if token:
        headers['Authorization'] = f'token {token}'

    response = requests.get(f'https://api.github.com/repos/{user}/{repo}/branches', headers=headers)

    branches = [branch['name'] for branch in response.json()]
    return branches


def generate_fake_responses(documents: List[Document]) -> Dict[str, str]:
    responses = {}
    import random
    for doc in documents:
        var = 'far' if random.randint(1,100) > 50 else 'bar'
        notebook_name = doc.metadata['source'].split('/')[-1]
        responses[notebook_name] = f'\n - `{notebook_name}`: This notebooks is used for {var}'
    return responses


def retrieve_summary(documents: List[Document], embeddings: Embeddings, context: Optional[str]='',
                      model_type: str='FakeLLM', chunk_size: int=2048,
                      chunk_overlap: int=128, print_token_n_costs: bool=False) -> Tuple[BaseLLM,  Dict[str, Dict[str, str]]]:
    
    total_cost = 0
    results_parent_folder_dict={}
    if model_type == 'GPT4ALL':
        callbacks = [StreamingStdOutCallbackHandler()]
        llm = GPT4All(model=model_name, temp=0, callbacks=callbacks, verbose=False)
        chunk_size = 500
        chunk_overlap = 50
    elif model_type == 'OpenAI':
        llm = OpenAI(temperature=0, max_tokens=500)
    elif model_type == 'FakeLLM':
        responses = generate_fake_responses(documents)
        llm = FakeListLLM(responses=list(responses.values()))
    else:
        raise ValueError('incorrect model_type (only supports OpenAI|GPT4ALL|FakeLLM))')
    for doc in documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                chunk_overlap=chunk_overlap,
                                                separators =  ["\n\n", "\n"])
        doc_split = text_splitter.split_documents([doc])
        search_index = Chroma.from_documents(doc_split, embeddings,)
        retriever = search_index.as_retriever()
        notebook_name = doc.metadata['source'].split('/')[-1]
        parent_folder = doc.metadata['parent_folder']
        if context:
            context = f"\n{context}\n"
        if parent_folder not in results_parent_folder_dict.keys():
              results_parent_folder_dict[parent_folder] = {}
        query = f"The following code comes from the {notebook_name} Jupyter Notebook. Your task is to explain for what this code is used for. {context}\
            Describe this using just one paragraph, including the location of the input and output files from the notebook if applicable. \
            Use a bullet point for your response, which must be in markdown. You must dont add more information. Here is the initial part of your answer:\
            - `{notebook_name}`: "
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        print(f'Working on {parent_folder}/{notebook_name}..')
        if model_type=='OpenAI' and print_token_n_costs:
            with get_openai_callback() as cb:
                res = qa(query)
                print(cb)
                total_cost += cb.total_cost
        elif model_type == 'FakeLLM':
            res = {'result' : responses[notebook_name]}
        else:
            res = qa(query)

        results_parent_folder_dict[parent_folder][notebook_name] = res['result']
        
    if total_cost > 0:
        print(f'Final cost (USD): ${total_cost}\n')
    return llm, results_parent_folder_dict

def summary_repo(llm: BaseLLM, summary_notebooks: str, repo_name: str,
                  model_type: str='FakeLLM', print_token_n_costs: bool=False) -> str:

    query_repo = f"""The following is the summary of a list of notebooks used in a GitHub repository called `{repo_name}`:
{summary_notebooks}
What do you think is this repo used for? Don't explain me each notebook, just give me a summary of the repository that could be added to a readme.md file."""

    if model_type=='OpenAI' and print_token_n_costs:
        with get_openai_callback() as cb:
            summary_repo = llm(query_repo)
            print(cb)
    elif model_type == 'FakeLLM':
            summary_repo = 'This repo contains notebooks for foo'
    else:
        summary_repo = llm(query_repo)
    return summary_repo