# Linear and logistic regression

## Extract sample size for each haplogroup across populations
### Mother
```r
df=read.table("Metadata.M.Final.tsv",header=TRUE,sep = '\t',quote="")

print("All Haplogroup breakdown")
table(df$MainHap)

print("African Haplogroup breakdown")
table(df[df$population=="African",]$MainHap)

print("South Asian Haplogroup breakdown")
table(df[df$population=="SouthAsian",]$MainHap)
```
```
[1] "All Haplogroup breakdown"

   A    B    C    D    E    F    G    H   HV    I    J    K   L0   L1   L2   L3
  68   13   20  146   59   84   62  103   20   10   60   42  672  380 1110 2049
  L4   L5    M    N    R    T    U    W    X    Z
 143   10 3355   42  359  106  628  131   24   28
[1] "African Haplogroup breakdown"

   B    E    I   L0   L1   L2   L3   L4   L5    M    R    T    U
   9   59    1  667  376 1100 2043  139   10   52    1   24   22
[1] "South Asian Haplogroup breakdown"

   A    B    C    D    F    G    H   HV    I    J    K   L0   L1   L2   L3   L4
  68    4   20  146   84   62  103   20    9   60   42    5    4   10    6    4
   M    N    R    T    U    W    X    Z
3303   42  358   82  606  131   24   28
```





### Child


```r
df=read.table("Metadata.C.Final.tsv",header=TRUE,sep = '\t',quote="")

print("All Haplogroup breakdown")
table(df$MainHap)

print("African Haplogroup breakdown")
table(df[df$population=="African",]$MainHap)

print("South Asian Haplogroup breakdown")
table(df[df$population=="SouthAsian",]$MainHap)
```
```
[1] "All Haplogroup breakdown"

   A    D    E    F    G    H   HV    J    K   L0   L1   L2   L3   L4    M    N
  29   82   47   32   38   58   15   32   26  428  175  769 1501  123 1712   27
   R    T    U    W    X    Z
 188   57  314   78   11   19
[1] "African Haplogroup breakdown"

   E   L0   L1   L2   L3   L4    M    R    T    U
  47  425  171  767 1498  118   35    1   16   20
[1] "South Asian Haplogroup breakdown"

   A    D    F    G    H   HV    J    K   L0   L1   L2   L3   L4    M    N    R
  29   82   32   38   58   15   32   26    3    4    2    3    5 1677   27  187
   T    U    W    X    Z
  41  294   78   11   19
```





# Use regression models to see population association with gestational age and pre-term birth (PTB)


```r
 Rscript reg.r > out.txt
```
