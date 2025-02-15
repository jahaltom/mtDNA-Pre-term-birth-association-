# Initial EDA.

For each categorical variable class, determine the number of pre-term births and normal births (PTB=1 normal=0) and the % of PTB=1. 
Missing data is removed for each variable one at a time.

```python
import pandas as pd

df = pd.read_csv("Metadata.M.Final.tsv", sep='\t')

# Columns to analyze
columns = ['TYP_HOUSE', 'HH_ELECTRICITY', 'FUEL_FOR_COOK', 'DRINKING_SOURCE',
                       'TOILET', 'WEALTH_INDEX', 'PASSIVE_SMOK', 'CHRON_HTN',
                       'DIABETES', 'TB', 'THYROID', 'EPILEPSY', 'BABY_SEX', 'MainHap',
                       "SNIFF_TOBA", "SMOKE_HIST"]
df=df[columns+["PTB"]]


# List to store results
results = []

# Loop through each column
for col in columns:
    dfT = df[~df[col].isin([-88, -77])]  # Remove rows with invalid entries (-88, -77)
    unique_values = dfT[col].drop_duplicates().to_list()
    for value in unique_values:
        # Calculate counts
        ptb_counts = dfT[dfT[col] == value]["PTB"].value_counts()
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
0         TYP_HOUSE     1         4173          334          7.410694
1         TYP_HOUSE     2         3605          265          6.847545
2    HH_ELECTRICITY     1         5226          461          8.106207
3    HH_ELECTRICITY     0         3154          182          5.455635
4     FUEL_FOR_COOK     1         1449          174         10.720887
5     FUEL_FOR_COOK     4         2988          280          8.567931
6     FUEL_FOR_COOK     3         3081          137          4.257303
7     FUEL_FOR_COOK     5          650           43          6.204906
8     FUEL_FOR_COOK     2          209            7          3.240741
9   DRINKING_SOURCE     1         4410          289          6.150245
10  DRINKING_SOURCE     2         2912          281          8.800501
11  DRINKING_SOURCE     4          451           29          6.041667
12  DRINKING_SOURCE     3            2            0          0.000000
13           TOILET     1         4420          430          8.865979
14           TOILET     2         2683          130          4.621401
15           TOILET     3         1261           82          6.105733
16           TOILET     4           14            1          6.666667
17     WEALTH_INDEX     5         1689          125          6.890849
18     WEALTH_INDEX     1         1464          104          6.632653
19     WEALTH_INDEX     3         1532          126          7.599517
20     WEALTH_INDEX     2         1518           93          5.772812
21     WEALTH_INDEX     4         1572          151          8.763784
22     PASSIVE_SMOK     1         2685          200          6.932409
23     PASSIVE_SMOK     0         4442          342          7.148829
24        CHRON_HTN     0         8167          611          6.960583
25        CHRON_HTN     1          206           29         12.340426
26         DIABETES     0         8349          632          7.037078
27         DIABETES     1           34           11         24.444444
28               TB     0         8302          638          7.136465
29               TB     1           72            4          5.263158
30          THYROID     0         8358          637          7.081712
31          THYROID     1           23            6         20.689655
32         EPILEPSY     0         8371          643          7.133348
33         EPILEPSY     1           12            0          0.000000
34         BABY_SEX     1         4222          334          7.330992
35         BABY_SEX     2         4132          307          6.915972
36          MainHap     M         2919          317          9.796044
37          MainHap     U          537           59          9.899329
38          MainHap     R          330           23          6.515581
39          MainHap     N           39            3          7.142857
40          MainHap     D          119           16         11.851852
41          MainHap     T           96           10          9.433962
42          MainHap     F           76            5          6.172840
43          MainHap     J           52            5          8.771930
44          MainHap     H           90            9          9.090909
45          MainHap     W          114           11          8.800000
46          MainHap     G           58            1          1.694915
47          MainHap     Z           25            3         10.714286
48          MainHap     K           34            7         17.073171
49          MainHap     A           60            3          4.761905
50          MainHap    L1          330           14          4.069767
51          MainHap    L2          960           55          5.418719
52          MainHap    L3         1776           72          3.896104
53          MainHap    L0          593           20          3.262643
54          MainHap    L4          125            7          5.303030
55          MainHap     E           50            3          5.660377
56       SNIFF_TOBA     1         6723          500          6.922331
57       SNIFF_TOBA     4          572           54          8.626198
58       SNIFF_TOBA     2           24            0          0.000000
59       SNIFF_TOBA     3           14            3         17.647059
60       SMOKE_HIST     1         7659          578          7.017118
61       SMOKE_HIST     4           16            1          5.882353
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
          
- Output results:Separate Bonferroni correction for each test type (Categorical_Analysis_Results.csv).
  
- Subsets df to specific columns and removes missing data across the board. 
- VIF is used to asses each variable for multicollinearity. MainHap and FUEL_FOR_COOK are one-hot encoded and the 1st is dropped. Outputs results in Categorical_Multicollinearity_VIF.csv.
- Pearson correlation: Using same df from above except 1st is not droped.

  




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



