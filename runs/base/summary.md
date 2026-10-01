## Primary model summary

| model_name   | dataset      |   num_runs |   mean_accuracy |   sample_variance |   std_dev |   std_error |   ci95_low |   ci95_high |   min_accuracy |   max_accuracy |
|:-------------|:-------------|-----------:|----------------:|------------------:|----------:|------------:|-----------:|------------:|---------------:|---------------:|
| qwenbase     | MATH500      |         10 |         69.36   |            1.136  |    1.0658 |      0.337  |    68.5975 |     70.1225 |        68      |        71.4    |
| qwenbase     | AIME24       |         10 |         10.3333 |            8.5185 |    2.9187 |      0.923  |     8.2455 |     12.4212 |         6.6667 |        16.6667 |
| qwenbase     | AIME25       |         10 |          7.3333 |           24.1975 |    4.9191 |      1.5556 |     3.8144 |     10.8522 |         0      |        13.3333 |
| qwenbase     | AMC23        |         10 |         46.25   |           36.4583 |    6.0381 |      1.9094 |    41.9306 |     50.5694 |        35      |        52.5    |
| qwenbase     | MacroAverage |         10 |         33.3192 |            3.177  |    1.7824 |      0.5636 |    32.0441 |     34.5942 |        30.1167 |        36.0333 |

## Primary run metrics

| model_name   |   run | dataset      |   correct |   total |   accuracy |
|:-------------|------:|:-------------|----------:|--------:|-----------:|
| qwenbase     |     0 | MATH500      |       345 |     500 |   69       |
| qwenbase     |     0 | AIME24       |         3 |      30 |   10       |
| qwenbase     |     0 | AIME25       |         4 |      30 |   13.3333  |
| qwenbase     |     0 | AMC23        |        16 |      40 |   40       |
| qwenbase     |     0 | MacroAverage |       nan |     nan |   33.0833  |
| qwenbase     |     1 | MATH500      |       354 |     500 |   70.8     |
| qwenbase     |     1 | AIME24       |         5 |      30 |   16.6667  |
| qwenbase     |     1 | AIME25       |         2 |      30 |    6.66667 |
| qwenbase     |     1 | AMC23        |        20 |      40 |   50       |
| qwenbase     |     1 | MacroAverage |       nan |     nan |   36.0333  |
| qwenbase     |     2 | MATH500      |       347 |     500 |   69.4     |
| qwenbase     |     2 | AIME24       |         3 |      30 |   10       |
| qwenbase     |     2 | AIME25       |         3 |      30 |   10       |
| qwenbase     |     2 | AMC23        |        21 |      40 |   52.5     |
| qwenbase     |     2 | MacroAverage |       nan |     nan |   35.475   |
| qwenbase     |     3 | MATH500      |       357 |     500 |   71.4     |
| qwenbase     |     3 | AIME24       |         3 |      30 |   10       |
| qwenbase     |     3 | AIME25       |         2 |      30 |    6.66667 |
| qwenbase     |     3 | AMC23        |        19 |      40 |   47.5     |
| qwenbase     |     3 | MacroAverage |       nan |     nan |   33.8917  |
| qwenbase     |     4 | MATH500      |       349 |     500 |   69.8     |
| qwenbase     |     4 | AIME24       |         3 |      30 |   10       |
| qwenbase     |     4 | AIME25       |         3 |      30 |   10       |
| qwenbase     |     4 | AMC23        |        16 |      40 |   40       |
| qwenbase     |     4 | MacroAverage |       nan |     nan |   32.45    |
| qwenbase     |     5 | MATH500      |       344 |     500 |   68.8     |
| qwenbase     |     5 | AIME24       |         4 |      30 |   13.3333  |
| qwenbase     |     5 | AIME25       |         1 |      30 |    3.33333 |
| qwenbase     |     5 | AMC23        |        14 |      40 |   35       |
| qwenbase     |     5 | MacroAverage |       nan |     nan |   30.1167  |
| qwenbase     |     6 | MATH500      |       345 |     500 |   69       |
| qwenbase     |     6 | AIME24       |         2 |      30 |    6.66667 |
| qwenbase     |     6 | AIME25       |         0 |      30 |    0       |
| qwenbase     |     6 | AMC23        |        20 |      40 |   50       |
| qwenbase     |     6 | MacroAverage |       nan |     nan |   31.4167  |
| qwenbase     |     7 | MATH500      |       341 |     500 |   68.2     |
| qwenbase     |     7 | AIME24       |         3 |      30 |   10       |
| qwenbase     |     7 | AIME25       |         4 |      30 |   13.3333  |
| qwenbase     |     7 | AMC23        |        18 |      40 |   45       |
| qwenbase     |     7 | MacroAverage |       nan |     nan |   34.1333  |
| qwenbase     |     8 | MATH500      |       346 |     500 |   69.2     |
| qwenbase     |     8 | AIME24       |         2 |      30 |    6.66667 |
| qwenbase     |     8 | AIME25       |         3 |      30 |   10       |
| qwenbase     |     8 | AMC23        |        20 |      40 |   50       |
| qwenbase     |     8 | MacroAverage |       nan |     nan |   33.9667  |
| qwenbase     |     9 | MATH500      |       340 |     500 |   68       |
| qwenbase     |     9 | AIME24       |         3 |      30 |   10       |
| qwenbase     |     9 | AIME25       |         0 |      30 |    0       |
| qwenbase     |     9 | AMC23        |        21 |      40 |   52.5     |
| qwenbase     |     9 | MacroAverage |       nan |     nan |   32.625   |

## Paired significance tests

| status       | reason                                                 |
|:-------------|:-------------------------------------------------------|
| not_computed | Set --baseline_model to run paired significance tests. |

