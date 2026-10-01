## Primary model summary

| model_name   | dataset      |   num_runs |   mean_accuracy |   sample_variance |   std_dev |   std_error |   ci95_low |   ci95_high |   min_accuracy |   max_accuracy |
|:-------------|:-------------|-----------:|----------------:|------------------:|----------:|------------:|-----------:|------------:|---------------:|---------------:|
| ours         | MATH500      |         10 |         71.16   |            0.5493 |    0.7412 |      0.2344 |    70.6298 |     71.6902 |        70      |        72.4    |
| ours         | AIME24       |         10 |         12.3333 |           10      |    3.1623 |      1      |    10.0712 |     14.5955 |        10      |        20      |
| ours         | AIME25       |         10 |          7.6667 |           14.9383 |    3.865  |      1.2222 |     4.9018 |     10.4315 |         3.3333 |        13.3333 |
| ours         | AMC23        |         10 |         49.75   |           40.9028 |    6.3955 |      2.0224 |    45.1749 |     54.3251 |        40      |        57.5    |
| ours         | MacroAverage |         10 |         35.2275 |            4.9705 |    2.2295 |      0.705  |    33.6326 |     36.8224 |        31.2833 |        37.9583 |

## Primary run metrics

| model_name   |   run | dataset      |   correct |   total |   accuracy |
|:-------------|------:|:-------------|----------:|--------:|-----------:|
| ours         |     0 | MATH500      |       359 |     500 |   71.8     |
| ours         |     0 | AIME24       |         3 |      30 |   10       |
| ours         |     0 | AIME25       |         1 |      30 |    3.33333 |
| ours         |     0 | AMC23        |        16 |      40 |   40       |
| ours         |     0 | MacroAverage |       nan |     nan |   31.2833  |
| ours         |     1 | MATH500      |       354 |     500 |   70.8     |
| ours         |     1 | AIME24       |         4 |      30 |   13.3333  |
| ours         |     1 | AIME25       |         2 |      30 |    6.66667 |
| ours         |     1 | AMC23        |        21 |      40 |   52.5     |
| ours         |     1 | MacroAverage |       nan |     nan |   35.825   |
| ours         |     2 | MATH500      |       360 |     500 |   72       |
| ours         |     2 | AIME24       |         4 |      30 |   13.3333  |
| ours         |     2 | AIME25       |         4 |      30 |   13.3333  |
| ours         |     2 | AMC23        |        17 |      40 |   42.5     |
| ours         |     2 | MacroAverage |       nan |     nan |   35.2917  |
| ours         |     3 | MATH500      |       355 |     500 |   71       |
| ours         |     3 | AIME24       |         3 |      30 |   10       |
| ours         |     3 | AIME25       |         1 |      30 |    3.33333 |
| ours         |     3 | AMC23        |        22 |      40 |   55       |
| ours         |     3 | MacroAverage |       nan |     nan |   34.8333  |
| ours         |     4 | MATH500      |       352 |     500 |   70.4     |
| ours         |     4 | AIME24       |         4 |      30 |   13.3333  |
| ours         |     4 | AIME25       |         1 |      30 |    3.33333 |
| ours         |     4 | AMC23        |        16 |      40 |   40       |
| ours         |     4 | MacroAverage |       nan |     nan |   31.7667  |
| ours         |     5 | MATH500      |       354 |     500 |   70.8     |
| ours         |     5 | AIME24       |         6 |      30 |   20       |
| ours         |     5 | AIME25       |         2 |      30 |    6.66667 |
| ours         |     5 | AMC23        |        21 |      40 |   52.5     |
| ours         |     5 | MacroAverage |       nan |     nan |   37.4917  |
| ours         |     6 | MATH500      |       357 |     500 |   71.4     |
| ours         |     6 | AIME24       |         3 |      30 |   10       |
| ours         |     6 | AIME25       |         3 |      30 |   10       |
| ours         |     6 | AMC23        |        21 |      40 |   52.5     |
| ours         |     6 | MacroAverage |       nan |     nan |   35.975   |
| ours         |     7 | MATH500      |       362 |     500 |   72.4     |
| ours         |     7 | AIME24       |         3 |      30 |   10       |
| ours         |     7 | AIME25       |         4 |      30 |   13.3333  |
| ours         |     7 | AMC23        |        21 |      40 |   52.5     |
| ours         |     7 | MacroAverage |       nan |     nan |   37.0583  |
| ours         |     8 | MATH500      |       355 |     500 |   71       |
| ours         |     8 | AIME24       |         4 |      30 |   13.3333  |
| ours         |     8 | AIME25       |         3 |      30 |   10       |
| ours         |     8 | AMC23        |        23 |      40 |   57.5     |
| ours         |     8 | MacroAverage |       nan |     nan |   37.9583  |
| ours         |     9 | MATH500      |       350 |     500 |   70       |
| ours         |     9 | AIME24       |         3 |      30 |   10       |
| ours         |     9 | AIME25       |         2 |      30 |    6.66667 |
| ours         |     9 | AMC23        |        21 |      40 |   52.5     |
| ours         |     9 | MacroAverage |       nan |     nan |   34.7917  |

## Paired significance tests

| status       | reason                                                 |
|:-------------|:-------------------------------------------------------|
| not_computed | Set --baseline_model to run paired significance tests. |

