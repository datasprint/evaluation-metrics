# Similarity

Symbols:
-  BK - Banking
-  LC - Lending & Credit
-  IR - Individuals Reference Data
-  ER - Entities Reference Data
-  TC - Telecommunications Data
-  DD - Device Data


### KDE Similarity Metric
**Applicable:**

BK, LC, IR, ER

**Explanation:**

Kernel Density Estimation (KDE) is a way of obtaining an estimated probability density function from a random variable. The curves of two KDE plots can be compared with one another. There are a few metrics that can be used to compare curves. Among others you one can use the curve length difference, partial curve mapping, discrete Frechet distance, dynamic time warping, and the area between curves. This technique would work with any other curves, like ROCAUC curves, or cumulative sum curves. 

**Output:**

```
{'Area Between Curves': 0.60937,
 'Curve Length Difference': 25.05818,
 'Discrete Frechet Distance': 2.04475,
 'Dynamic Time Warping': 210.27023,
 'Mean Absolute Difference': 0.52755,
 'Partial Curve Mapping': 157.43363}
 ```

**Visualisation:**
![](assets/kde-curve-date.png)

