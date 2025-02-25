# Initial EDA.

For each categorical variable class, determine the number of pre-term births and normal births (PTB=1 normal=0) and the % of PTB=1. 
Missing data is removed for each variable one at a time.

```python
import pandas as pd

df = pd.read_csv("Metadata.M.Final.tsv", sep='\t')

# Columns to analyze
columns = [  'TYP_HOUSE', 'HH_ELECTRICITY', 'FUEL_FOR_COOK', 'DRINKING_SOURCE',
                             'TOILET', 'WEALTH_INDEX','CHRON_HTN',
                             'DIABETES', 'TB', 'THYROID', 'EPILEPSY', 'BABY_SEX', 'MainHap',
                             'SMOKE_HIST','SMOK_FREQ']
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
0         TYP_HOUSE     1         3961          307          7.193065
1         TYP_HOUSE     2         3290          240          6.798867
2    HH_ELECTRICITY     1         4309          381          8.123667
3    HH_ELECTRICITY     0         2942          166          5.341055
4     FUEL_FOR_COOK     1         1258          154         10.906516
5     FUEL_FOR_COOK     4         2825          259          8.398184
6     FUEL_FOR_COOK     3         2614          105          3.861714
7     FUEL_FOR_COOK     5          358           22          5.789474
8     FUEL_FOR_COOK     2          196            7          3.448276
9   DRINKING_SOURCE     1         4059          260          6.019912
10  DRINKING_SOURCE     2         2771          259          8.547855
11  DRINKING_SOURCE     4          420           28          6.250000
12  DRINKING_SOURCE     3            1            0          0.000000
13           TOILET     1         3765          360          8.727273
14           TOILET     2         2280          110          4.602510
15           TOILET     3         1193           76          5.988968
16           TOILET     4           13            1          7.142857
17     WEALTH_INDEX     5         1583          120          7.046389
18     WEALTH_INDEX     1         1364           96          6.575342
19     WEALTH_INDEX     3         1439          114          7.340631
20     WEALTH_INDEX     2         1412           83          5.551839
21     WEALTH_INDEX     4         1453          134          8.443604
22        CHRON_HTN     0         7071          523          6.887016
23        CHRON_HTN     1          180           24         11.764706
24         DIABETES     0         7219          538          6.935671
25         DIABETES     1           32            9         21.951220
26               TB     0         7213          545          7.025006
27               TB     1           38            2          5.000000
28          THYROID     0         7229          542          6.974649
29          THYROID     1           22            5         18.518519
30         EPILEPSY     0         7239          547          7.025430
31         EPILEPSY     1           12            0          0.000000
32         BABY_SEX     1         3675          289          7.290616
33         BABY_SEX     2         3576          258          6.729264
34          MainHap     M         2692          288          9.664430
35          MainHap     U          503           55          9.856631
36          MainHap     R          305           20          6.153846
37          MainHap     N           37            3          7.500000
38          MainHap     D          112           13         10.400000
39          MainHap     T           87           10         10.309278
40          MainHap     F           70            5          6.666667
41          MainHap     J           43            4          8.510638
42          MainHap     H           74            9         10.843373
43          MainHap     W          109           11          9.166667
44          MainHap     G           56            1          1.754386
45          MainHap     Z           23            2          8.000000
46          MainHap     K           32            7         17.948718
47          MainHap     A           57            3          5.000000
48          MainHap    L1          203            5          2.403846
49          MainHap    L2          768           39          4.832714
50          MainHap    L3         1465           53          3.491436
51          MainHap    L0          451           11          2.380952
52          MainHap    L4          115            6          4.958678
53          MainHap     E           49            2          3.921569
54       SMOKE_HIST     1         7241          547          7.023626
55       SMOKE_HIST     4            8            0          0.000000
56       SMOKE_HIST     2            2            0          0.000000
57        SMOK_FREQ     0         7241          547          7.023626
58        SMOK_FREQ     1           10            0          0.000000

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

  

## Stat results (Separate Bonferroni correction for each test type)
```
            Variable            Test       P-Value Effect Size Significant
0         TYP_HOUSE            Chi2  5.261449e-01    0.007178       False
1         TYP_HOUSE           ANOVA  8.775989e-01        None       False
2         TYP_HOUSE  Kruskal-Wallis  7.388067e-01        None       False
3    HH_ELECTRICITY            Chi2  3.081018e-06    0.052831        True
4    HH_ELECTRICITY           ANOVA  2.006167e-14        None        True
5    HH_ELECTRICITY  Kruskal-Wallis  3.395675e-13        None        True
6     FUEL_FOR_COOK            Chi2  3.313437e-18    0.106299        True
7     FUEL_FOR_COOK           ANOVA  1.934785e-62        None        True
8     FUEL_FOR_COOK  Kruskal-Wallis  4.410464e-63        None        True
9   DRINKING_SOURCE            Chi2           NaN  Low counts         NaN
10  DRINKING_SOURCE           ANOVA           NaN        None         NaN
11  DRINKING_SOURCE  Kruskal-Wallis           NaN        None         NaN
12           TOILET            Chi2           NaN  Low counts         NaN
13           TOILET           ANOVA  2.434031e-22        None        True
14           TOILET  Kruskal-Wallis  9.006812e-21        None        True
15     WEALTH_INDEX            Chi2  3.197613e-02      0.0368       False
16     WEALTH_INDEX           ANOVA  8.633056e-02        None       False
17     WEALTH_INDEX  Kruskal-Wallis  1.286651e-01        None       False
18        CHRON_HTN            Chi2  1.067906e-02    0.028911       False
19        CHRON_HTN           ANOVA  2.769465e-01        None       False
20        CHRON_HTN  Kruskal-Wallis  5.792026e-01        None       False
21         DIABETES            Chi2  5.643900e-04    0.039048        True
22         DIABETES           ANOVA  9.474655e-03        None       False
23         DIABETES  Kruskal-Wallis  5.431536e-02        None       False
24               TB          Fisher  1.000000e+00        None       False
25               TB           ANOVA  2.657548e-01        None       False
26               TB  Kruskal-Wallis  2.447062e-01        None       False
27          THYROID            Chi2  4.916194e-02    0.022277       False
28          THYROID           ANOVA  9.452335e-02        None       False
29          THYROID  Kruskal-Wallis  1.072453e-01        None       False
30         EPILEPSY          Fisher  1.000000e+00        None       False
31         EPILEPSY           ANOVA  1.853433e-01        None       False
32         EPILEPSY  Kruskal-Wallis  1.961054e-01        None       False
33         BABY_SEX            Chi2  3.544465e-01    0.010486       False
34         BABY_SEX           ANOVA  1.258746e-02        None       False
35         BABY_SEX  Kruskal-Wallis  5.904382e-03        None       False
36          MainHap            Chi2           NaN  Low counts         NaN
37          MainHap           ANOVA  1.050111e-53        None        True
38          MainHap  Kruskal-Wallis  2.469218e-55        None        True
39       SMOKE_HIST            Chi2           NaN  Low counts         NaN
40       SMOKE_HIST           ANOVA  7.810725e-02        None       False
41       SMOKE_HIST  Kruskal-Wallis  2.324281e-02        None       False
42        SMOK_FREQ          Fisher  1.000000e+00        None       False
43        SMOK_FREQ           ANOVA  4.367043e-02        None       False
44        SMOK_FREQ  Kruskal-Wallis  9.874577e-03        None       False

```
## VIF results
```
            Variable          VIF
0         TYP_HOUSE  2641.378443
1    HH_ELECTRICITY     1.278385
2   DRINKING_SOURCE     1.807912
3            TOILET     2.024542
4      WEALTH_INDEX     1.481074
5         CHRON_HTN     1.458008
6          DIABETES     1.043827
7                TB     1.023076
8           THYROID     1.019314
9          EPILEPSY     1.008520
10         BABY_SEX     1.005745
11       SMOKE_HIST     1.004574
12        SMOK_FREQ    11.688740
13              PTB    11.711165
14         GAGEBRTH     1.747014
15  FUEL_FOR_COOK_2     1.790288
16  FUEL_FOR_COOK_3     2.390498
17  FUEL_FOR_COOK_4    11.965263
18  FUEL_FOR_COOK_5     3.154946
19        MainHap_D     3.041350
20        MainHap_E     3.045946
21        MainHap_F     2.157259
22        MainHap_G     2.239849
23        MainHap_H     1.941180
24        MainHap_J     2.414375
25        MainHap_K     1.797443
26       MainHap_L0     1.650344
27       MainHap_L1    10.848563
28       MainHap_L2     5.565465
29       MainHap_L3    17.415365
30       MainHap_L4    28.783636
31        MainHap_M     3.665280
32        MainHap_N    31.550164
33        MainHap_R     1.664170
34        MainHap_T     6.177904
35        MainHap_U     2.645051
36        MainHap_W     9.654132
37        MainHap_Z     2.968777
```


