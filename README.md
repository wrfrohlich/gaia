# Experiments

This project contains a set of routines for experimenting with the analysis and comparison of inertial data acquired from the GWalk wearables by BTS Engineering and kinematic data obtained from the GaitLab biomechanics laboratory. The codes were developed in Python 3.10.

We strongly recommend installing the packages from the requirements using the command:


```bash
pip install -r requirements.txt
```

Then, select the desired experiment routine:

```bash
python3.10 routine01.py
```

## Activity Schedule

1. Correlations:
    - [X] Raw data removing NaN data;
    - [X] Converting NaN data to mean;
    - [X] Normalized data;
    - [X] Filtering:
        - [X] Butterworth - Low-Pass;
        - [X] Butterworth - High-pass;
        - [X] Butterworth - Band-pass;
    - [X] Vectorization;
    - [X] Feature Engineering;

2. Clusters:
   - [X] PCA;
   - [X] t-SNE;
   - [X] K-Means;
   - [ ] DBSCAN;
   - [X] Silhouette Score;
   - [X] Davies-Bouldin Coeficient;
  
3. Machine Learning:
   - [X] Linear Regression;
   - [X] Random Forest;
        - [X] Important Features;

4. Deep Learning:
   - [ ] LSTM;

## Important Track Points

**Upper Trunk:**

```
"c7.X", "c7.Y", "c7.Z" (C7 cervical vertebra)
"r_should.X", "r_should.Y", "r_should.Z" (right shoulder)
"l_should.X", "l_should.Y", "l_should.Z" (left shoulder)
```

**Lower Trunk:**

```
"sacrum.X", "sacrum.Y", "sacrum.Z" (sacrum)
"r_asis.X", "r_asis.Y", "r_asis.Z" (right anterior superior iliac crest)
"l_asis.X", "l_asis.Y", "l_asis.Z" (left anterior superior iliac crest)
"MIDASIS.X", "MIDASIS.Y", "MIDASIS.Z" (midpoint between the anterior superior iliac crests)
```

**Legs:**

```
"r_knee 1.X", "r_knee 1.Y", "r_knee 1.Z" (right knee)
"l_knee 1.X", "l_knee 1.Y", "l_knee 1.Z" (left knee)
"r_mall.X", "r_mall.Y", "r_mall.Z" (right ankle)
"l_mall.X", "l_mall.Y", "l_mall.Z" (left ankle)
"r_heel.X", "r_heel.Y", "r_heel.Z" (right heel)
"l_heel.X", "l_heel.Y", "l_heel.Z" (left heel)
"r_met.X", "r_met.Y", "r_met.Z" (right metatarsal)
"l_met.X", "l_met.Y", "l_met.Z" (left metatarsal)
```

**Other Useful Points::**

```
"SHO.X", "SHO.Y", "SHO.Z" (average shoulder position)
"PO.X", "PO.Y", "PO.Z" (center of mass position)
```