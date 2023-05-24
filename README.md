# summaryRepoGPT
A program to summary the content of .ipynb files from a GitHub repo


```
usage: summary_repo.py [-h] [--local LOCAL] [--model MODEL]

A description

options:
  -h, --help            show this help message and exit
  --local LOCAL, -l LOCAL
                        If the repository has been already downloaded, you can pass the folder path
  --model MODEL, -m MODEL
                        To use a preferred model (OpenAI, FakeLLM for testing, GPT4ALL)
```

If no local repo is passed, then it will ask for a repo and branch.

-----
## Environment
Rename example.env to .env and edit the variables appropriately.

```
GITHUB_TOKEN: GitHub token used to access to private repos. 
OPENAI_API_KEY= OpenAI API key
GPT4_ALL_MODEL= Path to a GPT4ALL model (actualy is not working)
EMBEDDINGS_MODEL=all-MiniLM-L6-v2
```