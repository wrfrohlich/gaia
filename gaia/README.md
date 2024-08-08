# Experiments

## Steps

1. Correlations:
    - [ ] Raw data removing NaN data;
    - [ ] Converting NaN data to mean;
    - [ ] Normalized data;
    - [ ] Filtering:
        - [ ] Butterworth - Low-Pass;
        - [ ] Butterworth - High-pass;
        - [ ] Butterworth - Band-pass;
    - [ ] Vectorization;
    - [ ] Feature Engineering;

2. Clusters:
   - [ ] PCA;
   - [ ] t-SNE;
   - [ ] K-Means;
   - [ ] DBSCAN;
   - [ ] Silhouette Score;
   - [ ] Davies-Bouldin Coeficient;

## Important Track Points

**Tronco Superior:**

```
"c7.X", "c7.Y", "c7.Z" (C7 vértebra cervical)
"r_should.X", "r_should.Y", "r_should.Z" (ombro direito)
"l_should.X", "l_should.Y", "l_should.Z" (ombro esquerdo)
```

**Tronco Inferior:**

```
"sacrum.X", "sacrum.Y", "sacrum.Z" (sacro)
"r_asis.X", "r_asis.Y", "r_asis.Z" (crista ilíaca antero-superior direita)
"l_asis.X", "l_asis.Y", "l_asis.Z" (crista ilíaca antero-superior esquerda)
"MIDASIS.X", "MIDASIS.Y", "MIDASIS.Z" (ponto médio entre as cristas ilíacas antero-superiores)
```

**Pernas:**

```
"r_knee 1.X", "r_knee 1.Y", "r_knee 1.Z" (joelho direito)
"l_knee 1.X", "l_knee 1.Y", "l_knee 1.Z" (joelho esquerdo)
"r_mall.X", "r_mall.Y", "r_mall.Z" (tornozelo direito)
"l_mall.X", "l_mall.Y", "l_mall.Z" (tornozelo esquerdo)
"r_heel.X", "r_heel.Y", "r_heel.Z" (calcanhar direito)
"l_heel.X", "l_heel.Y", "l_heel.Z" (calcanhar esquerdo)
"r_met.X", "r_met.Y", "r_met.Z" (metatarso direito)
"l_met.X", "l_met.Y", "l_met.Z" (metatarso esquerdo)
```

**Outros pontos úteis:**

```
"SHO.X", "SHO.Y", "SHO.Z" (posição média dos ombros)
"PO.X", "PO.Y", "PO.Z" (posição do centro de massa)
```