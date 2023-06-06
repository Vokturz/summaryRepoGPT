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
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import (GPT4All, OpenAI, LlamaCpp)
from langchain.callbacks import get_openai_callback
from langchain.llms.fake import FakeListLLM
from langchain.embeddings.base import Embeddings
from langchain.llms.base import BaseLLM
from langchain import PromptTemplate
import tiktoken
import torch

load_dotenv()

model_name = os.environ.get("GPT4ALL_MODEL")
llama_model_path = os.environ.get("LLAMA_MODEL_PATH")

languages_dict = {
    'ipynb' : 'Jupyter Notebook',
    'py' : 'Python'
}

n_context_model = {
    'FakeLLM' : 4001,
    'OpenAI' : 4001,  # default davinci OpenAI
    'GPT4All' : 2048,
    'LlamaCpp' : 2048
}

def get_language_file(file_name):
    file_type = file_name.split('.')[-1]
    return languages_dict[file_type]


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


def load_notebook(file_path: str, include_markdown=True) -> Document:
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
            if include_markdown:
                lines = cell['source'].splitlines()
                commented_lines = ['# ' + line for line in lines]
                full_source.append('<markdown>\n'.join(commented_lines))
        else:
            full_source.append(cell['source'])
    full_source = '<code>\n'.join(full_source)
    doc = Document(page_content=full_source)
    doc.metadata['source'] = file_path
    parent_folder = '/'.join(file_path.split('/')[1:-1])
    if parent_folder == '':
        parent_folder = '.'
    doc.metadata['parent_folder'] = parent_folder
    return doc


def load_multiple_notebooks(source_dir: str, exclude_pattern: str='[!_]', include_markdown=True) -> List[Document]:
    """Convert all notebooks from a source_dir to langchain documents.
    It excludes all files starting with '_'.
    """
    ipynb_files = glob.glob(os.path.join(source_dir, f"{exclude_pattern}*.ipynb"), recursive=True)
    all_files = sorted(ipynb_files)
    return [load_notebook(file_path, include_markdown) for file_path in all_files]


def generate_fake_responses(documents: List[Document]) -> Dict[str, str]:
    """Generate fake responses for FakeLLM"""
    responses = {}
    import random
    for doc in documents:
        var = 'foo' if random.randint(1,100) > 50 else 'bar'
        notebook_name = doc.metadata['source'].split('/')[-1]
        responses[notebook_name] = f' is used for {var}.'
    return responses


def retrieve_summary(documents: List[Document], embeddings: Embeddings, extra_context: Optional[str]='',
                      model_type: str='FakeLLM', chunk_size: int=1500, chunk_overlap: int=0, n_ctx: Optional[int]=None,
                      n_threads: int=4, use_gpu: bool=False, max_tokens: int=300, chain_type: str='stuff', seed: Optional[int]=None,
                      show_spinner=True, print_token_n_costs: bool=False) -> Tuple[BaseLLM,  Dict[str, Dict[str, str]]]:
    """Obtain the summary of each document using the given LLM.

    It accepts an extra_context string to include extra information of the
    document (such as meaning of acronyms, etc).
    chunk_size and chunk_overlap are used to split the document.
    print_token_n_costs is used to include the number of tokens and price per
    query when using OpenAI.
    """
    total_cost = 0
    results_parent_folder_dict={}

    if chain_type == 'stuff':
        chunk_size = 450
    else:
        chunk_size = 4000
    # Get the prompt for code
    prompt, combine_prompt = prompt_by_function(model_type, "code", chain_type)

    if n_ctx is None:
        n_ctx = n_context_model[model_type]

    max_tokens_prompt = n_ctx - max_tokens
    
    print(f"Loading {model_type} model")
    # GPT4All models
    if model_type == 'GPT4All':
        #callbacks = [StreamingStdOutCallbackHandler()]
        llm = GPT4All(model=model_name, temp=0, n_predict=min([256,max_tokens]),
                      n_ctx=n_ctx, allow_download=True,
                      n_threads=n_threads, callbacks=None, verbose=False)
    # OpenAI model
    elif model_type == 'OpenAI':
        llm = OpenAI(temperature=0, max_tokens=max_tokens)
    # FakeLLM for testing purposes
    elif model_type == 'FakeLLM':
        pass
    # LlaMa models from LlamaCpp, it supports GPU usage
    elif model_type == 'LlamaCpp':
        n_gpu_layers = calculate_layer_count() if use_gpu else None
        llm = LlamaCpp(model_path=llama_model_path,  n_gpu_layers=n_gpu_layers,
                        temperature=0, max_tokens=max_tokens, n_ctx=n_ctx,
                        n_threads=n_threads,  callbacks=None, verbose=False)
        llm.client.verbose = False
    else:
        raise ValueError('Incorrect model_type (only supports OpenAI|GPT4All|LlamaCpp|FakeLLM))')
    
    # Split the documents
    for full_doc in documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                chunk_overlap=chunk_overlap,
                                                separators =  [ "\n\n", "\n", "<markdown>\n", "<code>\n", ""])
        doc_split = text_splitter.split_documents([full_doc])
        n_tokens =[]
        for doc in doc_split:
            # remove extra break lines
            doc.page_content = doc.page_content.replace("<markdown>\n", "\n") 
            doc.page_content = doc.page_content.replace("<code>\n", "\n") 
            doc.page_content = doc.page_content.replace("\n\n", "\n") 
            # count number of tokens
            n_tokens.append(num_tokens_from_string(doc.page_content, "p50k_base"))

        # For testing cases only
        if model_type == 'FakeLLM':
            responses = list(generate_fake_responses(doc_split).values())
            llm = FakeListLLM(responses=responses*100)

        # Document's metadata
        notebook_name = doc.metadata['source'].split('/')[-1]
        parent_folder = doc.metadata['parent_folder']

        # Add extra context
        if extra_context:
            extra_context = f"{extra_context}\n"

        if parent_folder not in results_parent_folder_dict.keys():
              results_parent_folder_dict[parent_folder] = {}

    
        file_type = get_language_file(notebook_name)   
        prompt_partial = prompt.partial(file_name=notebook_name, 
                                        file_type=file_type,
                                        extra_context=extra_context)
        
        extra_tokens = num_tokens_from_string(prompt_partial.format(context=''), "p50k_base") 
        extra_tokens += 150
        import random # Random until I find a better method :c
        random.seed(seed)
        k = len(doc_split)
        if k==1:
            total_tokens = n_tokens[0]
        elif k==2:
            total_tokens = n_tokens[0] + n_tokens[-1]
        else:
            random_chunks = sorted(random.sample(range(1,len(doc_split)-1),k-2))
            total_tokens = sum([n_tokens[0]] + [n_tokens[i] for i in random_chunks]  + [n_tokens[-1]])
            total_tokens += extra_tokens
        while total_tokens > max_tokens_prompt and k>1:
            random_chunks = sorted(random.sample(range(1,len(doc_split)-1),k-2))
            total_tokens = sum([n_tokens[0]] + [n_tokens[i] for i in random_chunks]  + [n_tokens[-1]])
            total_tokens += extra_tokens
            k-=1
        if chain_type == 'map_reduce':
            if k < len(doc_split):
                # Format the prompt
                combine_prompt_partial = combine_prompt.partial(file_name=notebook_name, 
                                            file_type=file_type)  
                chain = load_summarize_chain(llm, chain_type="map_reduce",
                                            map_prompt=prompt_partial,
                                            combine_prompt=combine_prompt_partial,
                                            combine_document_variable_name="context",
                                            map_reduce_document_variable_name="context")
            else: # use stuff
                print('Full file can be used as context, changing to chain_type=`stuff`')
                print(f'Using {k} chunks out of {len(doc_split)}. Total Tokens={total_tokens + max_tokens}')
                # Format the prompt
                prompt_partial = prompt.partial(file_name=notebook_name, 
                                        file_type=file_type,
                                        extra_context=extra_context)
                
                if k >2:
                    doc_split = [doc_split[0]] + [doc_split[i] for i in random_chunks]  + [doc_split[-1]]
                chain = load_summarize_chain(llm, chain_type='stuff',
                                            prompt=prompt_partial,
                                            document_variable_name="context")     

        else: # stuff     
            # Obtain best k for stuff
            #k = 0
            #sorted_n_tokens = sorted(n_tokens)[::-1]
            # total_tokens_worst_case = sorted_n_tokens[0]
            # total_tokens_worst_case += extra_tokens # to be safe
            # while total_tokens_worst_case<=max_tokens_prompt:
            #     k+=1 
            #     if k>=len(doc_split):
            #         break
            #     total_tokens_worst_case += sorted_n_tokens[k] 

            # # include another chunk
            # if k < len(doc_split): # total_tokens_worst_case + max_tokens < n_ctx 
            #     k+=1
            #     total_tokens_worst_case += max_tokens
            # # print(f'Chunk size={chunk_size}, Total chunks={len(doc_split)}')
            # print(f'Using {k} chunks out of {len(doc_split)}. Total Tokens (worst case)={total_tokens_worst_case}')

            # Create chromadb
            # search_index = Chroma.from_documents(documents=doc_split,
            #                                     embedding=embeddings,)
            # retriever = search_index.as_retriever(search_kwargs={"k": k})
            # doc_split = retriever.get_relevant_documents(f'What {notebook_name} does')


            print(f'Using {k} chunks out of {len(doc_split)}. Total Tokens={total_tokens + max_tokens}')
            doc_split = [doc_split[0]] + [doc_split[i] for i in random_chunks]  + [doc_split[-1]]
            chain = load_summarize_chain(llm, chain_type='stuff',
                                        prompt=prompt_partial,
                                        document_variable_name="context")

        # Start the query process
        if show_spinner:
            spinner = Halo(text=f'Working on {parent_folder}/{notebook_name}.. ({chain_type} mode)', spinner='dots')
            spinner.start()
        start_time = time.time()
        cb = None
        if model_type=='OpenAI' and print_token_n_costs:
            with get_openai_callback() as cb:
                res = chain({"input_documents" : doc_split}, return_only_outputs=True)
                total_cost += cb.total_cost
        else:
            res = chain({"input_documents" : doc_split}, return_only_outputs=True)
        if show_spinner:
            spinner.stop()

        output_var = 'output_text'
        res[output_var] = res[output_var].replace("\'", "'").replace("\\_", "_").strip()
        res[output_var] = f"\n- `{notebook_name}`: This {file_type} file {res[output_var]}"

        final_time = round((time.time()-start_time)/60,2) # as minutes
        print(f'Running time: {final_time} min')
        if cb:
            print(cb)

        # Should not happen
        if 'ERROR: The prompt size exceeds' in res[output_var]:
            print(res[output_var])
            exit()
        
        # Save results by parent_folder and notebook name
        results_parent_folder_dict[parent_folder][notebook_name] = res[output_var]

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
        summary_notebooks += f'**{parent_folder}** folder:'
        summary_notebooks += ''.join(llm_results) + '\n\n'
    return summary_notebooks


def summary_repo(llm: BaseLLM, summary_notebooks: str, repo_name: str,
                  model_type: str='FakeLLM', print_token_n_costs: bool=False) -> str:
    """Given a summary of notebooks, obtain a summary of the full repository

    print_token_n_costs is used to include the number of tokens and price of
    the query when using OpenAI.
    """
    # Get the prompt for the repository summary
    prompt_repo, _ =  prompt_by_function(model_type, "repo")
    prompt_repo = prompt_repo.format(repo_name=repo_name, 
                                     summary_notebooks=summary_notebooks)
    # Start query process
    spinner = Halo(text=f'Summarizing repo..', spinner='dots')
    spinner.start()
    cb = None
    if model_type == 'FakeLLM':
        llm = FakeListLLM(responses=['contains code for foo'])
    if model_type=='OpenAI' and print_token_n_costs:
        with get_openai_callback() as cb:
            summary_repo = llm(prompt_repo)
    else:
        summary_repo = llm(prompt_repo)
    spinner.stop()
    if cb:
        print(cb)

    summary_repo = "This repository " + summary_repo.strip()
    return summary_repo


def create_readme(repo_name: str, summary_notebooks: str, summary_repo: str) -> str:
    """Add the summary of the repo and notebooks to one string.
    The output looks like this:
    # <repo_name>
    <summary of the repository>

    Folders info:
    **<parent_folder>** folder:
        - 00_notebook.ipynb : ...
    and so on.
    """
    markdown_file = ""
    markdown_file += f"# {repo_name}\n"
    markdown_file += f'{summary_repo}\n\nFolders info:\n'
    markdown_file += f'{summary_notebooks}'
    return markdown_file



def prompt_by_function(model: str, to_summarize: str="code", chain_type: str="stuff") -> str:
    if chain_type=="stuff":
        if to_summarize == "code":
            prompt = (
                '\n\n"{context}"\n\n'
                "These are parts of the code from `{file_name}` {file_type} file. {extra_context}"
                "Your task is to understand and explain for what this code is used for, in a very general way."
                "Question: Conceptualize this code, making a concise summary of what it does."
                "Helpful Answer: This {file_type} file "
                )
            prompt = PromptTemplate(template=prompt, input_variables=["context", "file_name",
                                                                      "file_type", "extra_context"])
            return prompt, None
            
        
    elif chain_type=='map_reduce':
        if to_summarize == "code":
            map_prompt = (
                '\n\n"{context}"\n\n'
                "This code comes from `{file_name}` {file_type} file."
                "{extra_context}"
                "Question: Make a very concise summary of what it does, using no more than 50 words."
                "Concise summary: This {file_type} file "
                )
            
            combine_prompt = (
                "These are summaries of different parts of what the `{file_name}` {file_type} file does. This file:"
                '\n\n"{context}"\n\n'
                "Your task is to do a final summary of the file in just one paragraph"
                "Question: Conceptualize this code, making a concise summary of what it does."
                "Helpful Answer: This {file_type} file "
            )
            map_prompt = PromptTemplate(template=map_prompt,
                                        input_variables=["context", "file_name",
                                                         "file_type", "extra_context"])
            combine_prompt = PromptTemplate(template=combine_prompt,
                                            input_variables=["context",
                                                             "file_name",
                                                             "file_type"])
            return map_prompt, combine_prompt

    if to_summarize == "repo":
        prompt = (
            "This is a list of the summarized files from the GitHub repository called `{repo_name}`:"
            "{summary_notebooks}"
            "What do you think is this repo used for? Don't explain me each notebook, just give me a summary of the repository that could be added to a README.md file."
            "Concise summary: This repository "
            )
        prompt = PromptTemplate(template=prompt, input_variables=["repo_name", "summary_notebooks"])
        return prompt, None
        

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
    
