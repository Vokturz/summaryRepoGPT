```bash
python summary_notebook.py --file "repositories/data-science-ipython-notebooks/scipy/effect_size.ipynb" -m <model>
```

## OpenAI
**Running time**: 0.3 min
- `effect_size.ipynb`: This code is used to explore statistics that quantify effect size by looking at the difference in height between men and women. It takes data from the Behavioral Risk Factor Surveillance System (BRFSS) to estimate the mean and standard deviation of height in cm for adult women and men in the U.S. It then uses `scipy.stats.norm` to represent the distributions and the `eval_pdf` function to evaluate the normal (Gaussian) probability density function (PDF) within 4 standard deviations of the mean. It also includes a function called `overlap_superiority` which estimates overlap and superiority based on a sample, and a function called `plot_pdfs` which takes Cohen's d, plots normal distributions with the given effect size, and prints their overlap and superiority. Finally, it includes an interactive widget to visualize what different values of d mean.

## LlamaCpp: wizardLM-7B.ggmlv3.q4_1.bin
> Using `--n-threads 8` and GPU

**Running time** 1.11 min
- `effect_size.ipynb`: The code reads in data from a CSV file called "male_sample.csv" and "female_sample.csv", which contain mean heights for men and women, respectively. It then computes the difference between these means using the `scipy.stats` module, and plots the resulting normal distributions using the `matplotlib` library. Finally, it calculates the overlap and probability of superiority between the two distributions using the `overlap_superiority` function, and prints out the results. The code is located in the Jupyter Notebook file "effect_size.ipynb"

## GPT4All: ggml-gpt4all-j-v1.3-groovy
**Running time**: 16.64 min
- `effect_size.ipynb`:  This is an example Python script that uses NumPy to calculate overlap between two normal distributions with different standard deviations (cohden effect size). The function `plot_pdfs` takes in Cohen's $d$ as a parameter, plots PDF curves for control and treatment samples using the matplolib library. It then calculates their overlaps by summing up all values above or below each curve that are within 4 standard deviations of mean height (thresh). Finally it prints out both overlap and superiority statistics to help readers understand how Cohen's $d$ is calculated, as well as its interpretation in terms of effect size for a given population.

## GPT4All: ggml-vicuna-13b-1.1-q4_2
**Running time**: 22.11 min
- `effect_size.ipynb`:  This Python script calculates Cohen\'s effect size using two different methods to quantify differences between normal distributions with means 178 cm (for men) and 163 cm (for women). The first method is based on a threshold value of the difference in height, while the second uses probability density functions. It also includes an interactive widget that allows users to change Cohen\'s d and visualize its effect on the data. Finally, it calculates overlap and superiority statistics for different sample sizes using scipy libraries. The code is located within a file called "CompStats" in the directory where this text was saved.
