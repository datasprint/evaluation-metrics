# Utility

Symbols:
-  BK - Banking
-  LC - Lending & Credit
-  IR - Individuals Reference Data
-  ER - Entities Reference Data
-  TC - Telecommunications Data
-  DD - Device Data


### Averge Drop ROCAUC [[code](https://github.com/datasprint/evaluation-metrics/blob/master/utility/average-drop-rocauc.py), notebook]
**Applicable:**

BK, LC, IR, ER, DD

**Explanation:**

Calculate the predictability of all discrete features using LightGBM. Identify the average difference in predictability between the original and synthetic datasets. The target feature is being predicted using all other features excluding the target feature. 

**Output:**

```
![](https://docs.google.com/drawings/d/e/2PACX-1vSA1VOL5qRqRbpBB7rN3pFRNTbOTyN0OBCS6OcpPqzLXq-aSvSfPA3dAkg0Vr-3KMeneC7i9lZcYNft/pub?w=948&h=521)
 ```

**Visualisation:**
![](https://docs.google.com/drawings/d/e/2PACX-1vRP_NpXXuswdGtb7X6DWD0eeUCCratygZ31aUsn8iNZTVVs-T4msVFeuc7jyLbU9DIy71N5BULnOSqO/pub?w=896&h=613)



