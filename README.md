# summaryRepoGPT
This is a tool to summarize the content of .ipynb files from a GitHub repo using LLMs and  [`langchain`](https://github.com/hwchase17/langchain). It generates a markdown file `notebooks_summary.md` inside the repository folder containing a full report.

If you've ever found yourself lost within a labyrinth of notebooks, unsure of the core functionalities and overall purpose of a repo, this tool is here to provide clarity and facilitate understanding!


### Features 🚀
- **Summarization**: For each Jupyter notebook in a repository, the tool provides a succinct summary to help users quickly grasp its main functionality. Say goodbye to confusion and hello to clarity!

- **Repository Overview**: Beyond individual notebooks, the tool also offers an aggregated synopsis of the entire repository. No more endless scrolling through countless notebooks - get a comprehensive overview at a glance!

- **Cloning Capabilities**: This application goes beyond public repositories. With the right access token, it can clone and analyze even private repositories, while respecting their privacy settings. Your private repository remains private - the tool simply aids in the analysis and understanding of the code within your secure projects. This feature makes it easy to review code in a private setting, or dive into a team's internal projects without compromising privacy.

## Usage 💡
This repository contains two primary Python scripts, summary_repo.py and summary_notebook.py, designed to analyze and summarize Data Science repositories and individual Jupyter notebooks respectively.

### Repository Summarization: `summary_repo.py`
This script generates summaries of each Jupyter notebook within a chosen repository. Here is how to use it:
```
usage: summary_repo.py [-h] [--local LOCAL] [--model MODEL] [--n-threads N_THREADS] [--gpu | --no-gpu | -g]

A description

options:
  -h, --help            show this help message and exit
  --local LOCAL, -l LOCAL
                        If the repository has been already downloaded, you can pass the folder path
  --model MODEL, -m MODEL
                        To use a preferred model (OpenAI, FakeLLM for testing, GPT4All, LlamaCpp)
  --n-threads N_THREADS, -t N_THREADS
                        Number of threads to use
  --gpu, --no-gpu, -g   To run using GPU (Only for LlamaCpp)
```

"If no local repository is provided, it prompts the user to enter a repository and branch. All repositories are downloaded into the folder `repositories` folder.

By default the model runs with model `FakeLLM`, i.e., it generates a fake summary. To change that use the flag `--model` with models `OpenAI` or `GPT4All`. 

### Notebook Summarization: `summary_notebook.py`

This script generates a summary for an individual Jupyter notebook. Here is how to use it:
```
usage: summary_notebook.py [-h] --file FILE [--model MODEL] [--n-threads N_THREADS] [--gpu | --no-gpu | -g]

A description

options:
  -h, --help            show this help message and exit
  --file FILE, -f FILE  Notebook path
  --model MODEL, -m MODEL
                        To use a preferred model (OpenAI, FakeLLM for testing, GPT4All, LlamaCpp)
  --n-threads N_THREADS, -t N_THREADS
                        Number of threads to use
  --gpu, --no-gpu, -g   To run using GPU (Only for LlamaCpp)
```

## Enable GPU acceleration
Currently, the tool supports the package 'llama.cpp' running on GPU. However, by default, the package `llama-cpp-python` does not include GPU support. To activate it, it must be installed from the terminal as follows:
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip3 install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```
## Examples 🎯

### `summary_repo.py`
![Example](./example/example_usage.gif)

In this example, the repo [donnemartin/data-science-ipython-notebooks](https://github.com/donnemartin/data-science-ipython-notebooks) is cloned and then the `scipy` folder is selected to be summarized. Each notebook is summarized using this structure:
- `00_notebook0.ipynb`: This notebook...
- `01_notebook1.ipynb`: This notebook...

After generating summaries for all notebooks, the script compiles a comprehensive report, providing an overview of the entire repository (or in this case, the selected folder). This summary report is saved as [`notebooks_summary.md`](./example/notebooks_summary.md).

### `summary_notebook.py`

For an example of using the `summary_notebook.py` script, refer to [`comparison_summary.md`](./example/comparison_summary.md). This file contains the results of summarizing the `scipy/effect_size.ipynb`notebook, again from the [donnemartin/data-science-ipython-notebooks](https://github.com/donnemartin/data-science-ipython-notebooks) repository.


-----
## Environment variables
Rename example.env to .env and edit the variables appropriately.

```
GITHUB_TOKEN: GitHub token used to access to private repos. 
OPENAI_API_KEY= OpenAI API key
GPT4ALL_MODEL= GPT4ALL model name (see https://gpt4all.io/models/models.json for all models)
EMBEDDINGS_MODEL=all-MiniLM-L6-v2
LLAMA_MODEL_PATH=models/wizardLM-7B.ggmlv3.q4_1.bin
GGML_CUDA_NO_PINNED=1
```

# To do list
- [x] Make GPT4ALL work
- [x] Improve notebook splitter
- [X] Add [llama.cpp](https://github.com/ggerganov/llama.cpp) support
- [ ] Add benchmark table of different models
- [ ] Add more files to summarize


## Possible problems
1. Cannot install requirements? Check that you have  `gcc 11` compiler. If not, here how to install it
```bash
sudo apt install build-essential manpages-dev software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update && sudo apt install gcc-11 g++-11
```