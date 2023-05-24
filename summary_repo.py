import inquirer
import requests
import argparse
import os
from dotenv import load_dotenv


import glob
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
import utils
load_dotenv()

embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL')


def user_repo_validation(answers, current):
    if '/' not in current:
        raise inquirer.errors.ValidationError("", reason="You must write the user and repo separated by '/'")
    current_split = current.split('/')
    if len(current_split) != 2:
        raise inquirer.errors.ValidationError("", reason="Incorrect format, there is more than one '/'")
    user, repo = current_split
    if len(user)==0: 
        raise inquirer.errors.ValidationError("", reason="user cannot be empty")
    elif len(repo)==0:
        raise inquirer.errors.ValidationError("", reason="repository cannot be empty")
    if github_token:
        headers = {'Authorization': f'token {github_token}'}
        response = requests.get(f'https://api.github.com/repos/{user}/{repo}/branches', headers=headers)
    
    else:    
        response = requests.get(f'https://github.com/{user}/{repo}')
    if response.status_code != 200:
        raise inquirer.errors.ValidationError("", reason="This repository does not exist or is private.\nCheck that the GitHub token is correctly defined")
    return True




def main():
    args = parse_arguments()

    if not args.local:
        repo_question = inquirer.Text('user_repo', message="GitHub repository (user/repo)", validate=user_repo_validation),             
        user_repo = inquirer.prompt(repo_question)['user_repo']
        user, repo = user_repo.split('/')
        
        branches = utils.get_branches(user, repo, github_token)
        default = ['main', 'master']
        questions = [inquirer.List(
            'branch',
            message="Select the branch that you want to summarize",
            choices=branches,
            default=default
        )]
        answers = inquirer.prompt(questions)
        branch = answers['branch']
        print(f'Clonning {user_repo} ({branch})..')
        repo_folder = utils.clone_repository(user, repo, branch, token=github_token)
    
    else:
        repo_folder = args.local
        
    if not os.path.exists(repo_folder):
        print(f'There is no {repo_folder} folder')


    source_directory = repo_folder
    repo_name = repo_folder.split('/')[-1]

    ipynb_files = glob.glob(os.path.join(repo_folder, f"**/[!_]*.ipynb"), recursive=True)
    parent_folders = ['/'.join(file_path.split('/')[1:-1]) for file_path in ipynb_files]
    unique_parent_folders, total_documents = np.unique(parent_folders, return_counts=True)
    unique_parent_folders = [pf.replace(f'{repo_name}/', '') for pf in unique_parent_folders]
    ipynb_count = []
    
    for p, t in zip(unique_parent_folders,total_documents):
       
        ipynb_count.append(f' {t} files from {p}')
    q = [inquirer.Checkbox("selected_parent_folders", message="Select .ipynb folder files to summarize",
                           choices=tuple(zip(ipynb_count, unique_parent_folders )),
                           default=unique_parent_folders)]
    answers = inquirer.prompt(q)
    selected_parent_folders = answers["selected_parent_folders"]
    if len(selected_parent_folders) == 0:
        print('No folder was selected')
        return
    documents = []
    for spf in selected_parent_folders:
        source_dir = f'{source_directory}/{spf}'
        documents.extend(utils.load_multiple_notebooks(source_dir))
    
    print(f"Loaded {len(documents)} documents from {source_directory}")

    print(f'Loading embeddings from {embeddings_model_name}..')
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    llm, results_pf = utils.retrieve_summary(documents, embeddings, 
                                             model_type=args.model, print_token_n_costs=True)
    summary_notebooks = utils.format_summary(results_pf, repo_name)
    
    print('Summarizing the repo..')
    summary_repo = utils.summary_repo(llm, summary_notebooks, repo_name, 
                                      model_type=args.model, print_token_n_costs=True)
    readme = utils.create_readme(repo_name, summary_notebooks, summary_repo)
    with open(f'{source_directory}/notebooks_summary.md', 'w') as f:
        f.write(readme)
    print(f'Markdown file saved on {source_directory}/notebooks_summary.md!')

def parse_arguments():
    parser = argparse.ArgumentParser(description='A description')
    parser.add_argument("--local", "-l", help='If the repository has been already downloaded, you can pass the folder path')
    parser.add_argument("--model", "-m", default='FakeLLM', help='To use a preferred model (OpenAI, FakeLLM for testing, GPT4ALL)')

    return parser.parse_args()

if __name__ == "__main__":
    github_token = os.environ.get("GITHUB_TOKEN")
    if not os.path.exists('repositories/'):
        os.makedirs('repositories/')
    main()
