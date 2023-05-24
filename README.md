# summaryRepoGPT
A program to summary the content of .ipynb files from a GitHub repo. It generates a markdown file `notebooks_summary.md` inside the repository folder.


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

If no local repo is passed, then it will ask for a repo and branch. All repo all download in the folder `repositories`.

By default the model runs with model `FakeLLM`, i.e., it generates a fake summary. To change that use the flag `--model OpenAI`. Currently `GPT4ALL` is not working

 -----
## Example
![Example](./example/example_usage.gif)

In this example the repo [donnemartin/data-science-ipython-notebooks](https://github.com/donnemartin/data-science-ipython-notebooks) is clonned and then the `scipy` folder is selected to summarize. Each notebooks is summarized as 
- `00_notebook0.ipynb`: This notebook...
- `01_notebook1.ipynb`: This notebook...

and then the program obtain a final explanation of the repo (or folder in this case) which is wrote in [`notebooks_summary.md`](./example/notebooks_summary.md) file.

-----
## Environment variables
Rename example.env to .env and edit the variables appropriately.

```
GITHUB_TOKEN: GitHub token used to access to private repos. 
OPENAI_API_KEY= OpenAI API key
GPT4_ALL_MODEL= Path to a GPT4ALL model (actualy is not working)
EMBEDDINGS_MODEL=all-MiniLM-L6-v2
```

# To do list
- [ ] Write comments explaining functions inside `utils`
- [ ] Improve splitter of notebooks (prioritizing split by cells is a good idea)
- [ ] Make GPT4ALL work -> it require reduce the size of the chunks in the splitter
- [ ] Add more files to summarize