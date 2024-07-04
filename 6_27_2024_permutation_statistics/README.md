# Permutation Statistics
In this assignment, you'll be using machine learning as a tool to detect and understand patterns in permutations. Data files have been generated for you in `data/permutations_{n}.csv` for $5 \le n \le 10$. In each data file there are all the permutations of length n and 9 associated statistics computed for each permutation.

Your goal is to train models and try to interpret them to guess at what the permutation statistics are. A template notebook is given to you in `template.ipynb` and you'll want to make a copy of it for each statistic you explore. The code is set up so that the model one-hot encodes each permutation and predicts the statistic just based off of the permutation. However, some of the statistics are correlated with one another, so you can also try modifying the models to one-hot encode the other statistics and use them as part of the input for training.

Most of the statistics are binary values that only take on 0 and 1, but there are some that take on wider range of values. This is summarized in the table below.

| Statistic | Range |
| :-- | :-- |
| `stat1` | $[0,1]$ |
| `stat2` | $[0,1]$ |
| `stat3` | $[0,1]$ |
| `stat4` | $[0,1]$ |
| `stat5` | $[0,\frac{n(n-1)}{2}]$ |
| `stat6` | $[0,1]$ |
| `stat7` | $[0,n-1]$ |
| `stat8` | $[0,1]$ |
| `stat9` | $[0,1]$ |

This assignment is hard: `stat1` and `stat2` are more straightforward and some of the others might not be amenable to understanding with our methods! You should have notebooks and made an attempt at each one and summarize your results in `FINDINGS.md`. Grades will be based on completed attempts at all 9 statistics + correct interpretations of the first 2. Each additional statistic correctly interpreted will be worth an additional 1 point of extra credit.