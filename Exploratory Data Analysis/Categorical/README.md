# Initial EDA.

For each categorical variable class, determine the number of pre-term births and normal births (PTB=1 normal=0) and the % of PTB=1

```python
import pandas as pd

df = pd.read_csv("Metadata.M.Final.tsv", sep='\t')

# Columns to analyze
columns = ['TYP_HOUSE', 'HH_ELECTRICITY', 'FUEL_FOR_COOK', 'DRINKING_SOURCE',
                       'TOILET', 'WEALTH_INDEX', 'PASSIVE_SMOK', 'CHRON_HTN',
                       'DIABETES', 'TB', 'THYROID', 'EPILEPSY', 'BABY_SEX', 'MainHap',
                       "SNIFF_TOBA", "SMOKE_HIST"]
df=df[columns+["PTB"]]
df = df[~df.isin([-88, -77]).any(axis=1)]  # Remove rows with invalid entries (-88, -77)

# List to store results
results = []

# Loop through each column
for col in columns:
    unique_values = df[col].drop_duplicates().to_list()
    for value in unique_values:
        # Calculate counts
        ptb_counts = df[df[col] == value]["PTB"].value_counts()
        # Ensure there are no missing categories (0 or 1)
        ptb_counts = ptb_counts.reindex([0, 1], fill_value=0)       
        # Calculate percentage of PTB = 1 relative to all
        percentage = (ptb_counts[1] / (ptb_counts[0] + ptb_counts[1] ) * 100) 
        # Append results as a row
        results.append({
            "Column": col,
            "Value": value,
            "PTB_0_Count": ptb_counts[0],
            "PTB_1_Count": ptb_counts[1],
            "PTB_1_Percentage": percentage
        })

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Display the table
print(results_df)
```

```
             Column Value  PTB_0_Count  PTB_1_Count  PTB_1_Percentage
0         TYP_HOUSE     1         3618          289          7.396980
1         TYP_HOUSE     2         3468          249          6.698951
2    HH_ELECTRICITY     0         2953          169          5.413197
3    HH_ELECTRICITY     1         4133          369          8.196357
4     FUEL_FOR_COOK     4         2758          250          8.311170
5     FUEL_FOR_COOK     1         1372          167         10.851202
6     FUEL_FOR_COOK     3         2590          104          3.860431
7     FUEL_FOR_COOK     5          160           10          5.882353
8     FUEL_FOR_COOK     2          206            7          3.286385
9   DRINKING_SOURCE     2         2655          247          8.511371
10  DRINKING_SOURCE     1         3983          262          6.171967
11  DRINKING_SOURCE     4          446           29          6.105263
12  DRINKING_SOURCE     3            2            0          0.000000
13           TOILET     1         3693          352          8.702101
14           TOILET     2         2194          108          4.691573
15           TOILET     3         1185           77          6.101426
16           TOILET     4           14            1          6.666667
17     WEALTH_INDEX     2         1386           86          5.842391
18     WEALTH_INDEX     5         1552          108          6.506024
19     WEALTH_INDEX     1         1327           94          6.615060
20     WEALTH_INDEX     4         1423          131          8.429858
21     WEALTH_INDEX     3         1398          119          7.844430
22     PASSIVE_SMOK     1         2670          198          6.903766
23     PASSIVE_SMOK     0         4416          340          7.148865
24        CHRON_HTN     0         6921          515          6.925767
25        CHRON_HTN     1          165           23         12.234043
26         DIABETES     0         7056          528          6.962025
27         DIABETES     1           30           10         25.000000
28               TB     0         7049          536          7.066579
29               TB     1           37            2          5.128205
30          THYROID     0         7064          533          7.015927
31          THYROID     1           22            5         18.518519
32         EPILEPSY     0         7074          538          7.067788
33         EPILEPSY     1           12            0          0.000000
34         BABY_SEX     1         3579          288          7.447634
35         BABY_SEX     2         3507          250          6.654245
36          MainHap     M         2711          294          9.783694
37          MainHap     G           55            1          1.785714
38          MainHap     W          103           11          9.649123
39          MainHap     U          511           50          8.912656
40          MainHap     R          312           19          5.740181
41          MainHap     F           68            5          6.849315
42          MainHap     D          113           13         10.317460
43          MainHap     N           36            3          7.692308
44          MainHap     Z           21            3         12.500000
45          MainHap     J           49            4          7.547170
46          MainHap     H           83            9          9.782609
47          MainHap     T           92           10          9.803922
48          MainHap     K           32            7         17.948718
49          MainHap     A           59            3          4.838710
50          MainHap    L1          143            5          3.378378
51          MainHap    L2          728           35          4.587156
52          MainHap    L3         1396           49          3.391003
53          MainHap    L0          409            8          1.918465
54          MainHap    L4          116            6          4.918033
55          MainHap     E           49            3          5.769231
56       SNIFF_TOBA     1         6521          484          6.909350
57       SNIFF_TOBA     4          528           51          8.808290
58       SNIFF_TOBA     2           23            0          0.000000
59       SNIFF_TOBA     3           14            3         17.647059
60       SMOKE_HIST     1         7067          538          7.074293
61       SMOKE_HIST     4           14            0          0.000000
62       SMOKE_HIST     2            4            0          0.000000
63       SMOKE_HIST     3            1            0          0.000000

```






# CategoricalEDA.py

- Takes in Metadata.M.Final.tsv 
- For each catigorical column, removes missing data.
    - For PTB; 
        - Performs Chi-Square and cramers v.
        
        - Fisher's Exact Test(if contingency_table.values < 5 and contingency_table.shape == (2, 2)  e.g. TB)(DRINKING_SOURCE has a value < 5 but not 2x2. This gets excluded). 
    - For GAGEBRTH;
        - ANOVA and Kruskal-Wallis

- Subsets df to specific columns and removes missing data across the board. 
- VIF is used to asses each variable for multicollinearity. MainHap and FUEL_FOR_COOK are one-hot encoded and the 1st is dropped. 
- Pearson correlation: Using same df from above except 1st is not droped.
- Output results:Separate Bonferroni correction for each test type.



## VIF results
```
           Variable          VIF
0         TYP_HOUSE  1492.133301
1    HH_ELECTRICITY     1.338904
2   DRINKING_SOURCE     1.933587
3            TOILET     2.070086
4      WEALTH_INDEX     1.532907
5      PASSIVE_SMOK     1.519001
6         CHRON_HTN     1.145544
7          DIABETES     1.040461
8                TB     1.023495
9           THYROID     1.007865
10         EPILEPSY     1.009333
11         BABY_SEX     1.006067
12       SNIFF_TOBA     1.004969
13       SMOKE_HIST     1.082677
14              PTB     1.008571
15         GAGEBRTH     1.733009
16  FUEL_FOR_COOK_2     1.782972
17  FUEL_FOR_COOK_3     2.437520
18  FUEL_FOR_COOK_4    11.640222
19  FUEL_FOR_COOK_5     3.093030
20        MainHap_D     1.711992
21        MainHap_E     2.991917
22        MainHap_F     2.149492
23        MainHap_G     2.167560
24        MainHap_H     1.894286
25        MainHap_J     2.512088
26        MainHap_K     1.868047
27       MainHap_L0     1.629379
28       MainHap_L1     9.678624
29       MainHap_L2     4.164354
30       MainHap_L3    16.140782
31       MainHap_L4    26.885350
32        MainHap_M     3.608939
33        MainHap_N    30.204153
34        MainHap_R     1.626251
35        MainHap_T     6.092557
36        MainHap_U     2.667763
37        MainHap_W     9.398611
38        MainHap_Z     2.810342


```
## Stat results (Separate Bonferroni correction for each test type)
```
           Variable            Test       P-Value Effect Size Significant
0         TYP_HOUSE            Chi2  3.396992e-01    0.010432       False
1         TYP_HOUSE           ANOVA  7.496995e-01        None       False
2         TYP_HOUSE  Kruskal-Wallis  8.048174e-01        None       False
3    HH_ELECTRICITY            Chi2  2.841368e-06    0.049289        True
4    HH_ELECTRICITY           ANOVA  5.688943e-13        None        True
5    HH_ELECTRICITY  Kruskal-Wallis  2.234656e-11        None        True
6     FUEL_FOR_COOK            Chi2  3.520700e-18    0.098778        True
7     FUEL_FOR_COOK           ANOVA  4.979702e-74        None        True
8     FUEL_FOR_COOK  Kruskal-Wallis  2.125231e-79        None        True
9   DRINKING_SOURCE            Chi2           NaN  Low counts         NaN
10  DRINKING_SOURCE           ANOVA  2.035932e-22        None        True
11  DRINKING_SOURCE  Kruskal-Wallis  7.173723e-26        None        True
12           TOILET            Chi2           NaN  Low counts         NaN
13           TOILET           ANOVA  1.499114e-25        None        True
14           TOILET  Kruskal-Wallis  3.995459e-23        None        True
15     WEALTH_INDEX            Chi2  1.296833e-02    0.038908       False
16     WEALTH_INDEX           ANOVA  1.155273e-01        None       False
17     WEALTH_INDEX  Kruskal-Wallis  2.155249e-01        None       False
18     PASSIVE_SMOK            Chi2  7.548443e-01    0.003566       False
19     PASSIVE_SMOK           ANOVA  1.289662e-01        None       False
20     PASSIVE_SMOK  Kruskal-Wallis  5.704247e-02        None       False
21        CHRON_HTN            Chi2  2.364234e-03    0.032023        True
22        CHRON_HTN           ANOVA  1.728665e-01        None       False
23        CHRON_HTN  Kruskal-Wallis  5.729685e-01        None       False
24         DIABETES            Chi2  2.256225e-05    0.044607        True
25         DIABETES           ANOVA  6.713991e-03        None       False
26         DIABETES  Kruskal-Wallis  4.846272e-02        None       False
27               TB          Fisher  6.583330e-01        None       False
28               TB           ANOVA  9.043271e-01        None       False
29               TB  Kruskal-Wallis  9.251903e-01        None       False
30          THYROID            Chi2  1.304457e-02    0.026134       False
31          THYROID           ANOVA  3.678319e-02        None       False
32          THYROID  Kruskal-Wallis  5.313253e-02        None       False
33         EPILEPSY          Fisher  1.000000e+00        None       False
34         EPILEPSY           ANOVA  2.127072e-01        None       False
35         EPILEPSY  Kruskal-Wallis  2.265318e-01        None       False
36         BABY_SEX            Chi2  4.690950e-01    0.007633       False
37         BABY_SEX           ANOVA  2.381770e-02        None       False
38         BABY_SEX  Kruskal-Wallis  1.065444e-02        None       False
39          MainHap            Chi2           NaN  Low counts         NaN
40          MainHap           ANOVA  4.789909e-67        None        True
41          MainHap  Kruskal-Wallis  1.243953e-73        None        True
42       SNIFF_TOBA            Chi2           NaN  Low counts         NaN
43       SNIFF_TOBA           ANOVA  1.176297e-04        None        True
44       SNIFF_TOBA  Kruskal-Wallis  5.462073e-05        None        True
45       SMOKE_HIST            Chi2           NaN  Low counts         NaN
46       SMOKE_HIST           ANOVA           NaN        None         NaN
47       SMOKE_HIST  Kruskal-Wallis           NaN        None         NaN
```



