
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




print("Population")
glm.fit=glm(GAGEBRTH_IN~population , data=df  ) 
print(summary (glm.fit ))
glm.fit=glm(PTB~population , family="binomial", data=df  )
print(summary (glm.fit ))


print("Site")
glm.fit=glm(GAGEBRTH_IN~site , data=df  ) 
print(summary (glm.fit ))
glm.fit=glm(PTB~site , family="binomial", data=df  )
print(summary (glm.fit ))
