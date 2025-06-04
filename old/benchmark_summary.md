
### Benchmark Results for cloud: **Lille_0** at dataset: **Paris_Lille**
  - **Operation**: neighSearch
  - **Kernel type**: Sphere
  - **Dataset size**: 10000000
  - **Average points found per search**: 50219
  - **Repeats**: 3
  - **With warmup**: Yes

| Method                                                  |   Runtime (Total) (ms) |   Mean Runtime (Per Search) (ms) | Improvement   |
|:--------------------------------------------------------|-----------------------:|---------------------------------:|:--------------|
| Pointer Octree, Unencoded                               |               2562.57  |                      0.000256257 | 1.00×         |
| Pointer Octree, MortonEncoder3D                         |               1813.14  |                      0.000181314 | 1.41×         |
| Pointer Octree, HilbertEncoder3D                        |               1484.83  |                      0.000148483 | 1.73×         |
| Pointer Octree, HilbertEncoder3D, Point + PointMetadata |               1276.27  |                      0.000127627 | 2.01×         |
| Linear Octree, MortonEncoder3D                          |               1731.65  |                      0.000173165 | 1.48×         |
| Linear Octree, HilbertEncoder3D                         |               1004.96  |                      0.000100496 | 2.55×         |
| Linear Octree, HilbertEncoder3D, Point + PointMetadata  |                849.284 |                      8.49284e-05 | 3.02×         |


### Benchmark Results for cloud: **Lille_0** at dataset: **Paris_Lille**
  - **Operation**: numNeighSearch
  - **Kernel type**: Sphere
  - **Dataset size**: 10000000
  - **Average points found per search**: 50219
  - **Repeats**: 3
  - **With warmup**: Yes

| Method                                                  |   Runtime (Total) (ms) |   Mean Runtime (Per Search) (ms) | Improvement   |
|:--------------------------------------------------------|-----------------------:|---------------------------------:|:--------------|
| Pointer Octree, Unencoded                               |               1940.83  |                      0.000194083 | 1.00×         |
| Pointer Octree, MortonEncoder3D                         |               1594.73  |                      0.000159473 | 1.22×         |
| Pointer Octree, HilbertEncoder3D                        |               1209.11  |                      0.000120911 | 1.61×         |
| Pointer Octree, HilbertEncoder3D, Point + PointMetadata |                999.489 |                      9.99489e-05 | 1.94×         |
| Linear Octree, MortonEncoder3D                          |                652.44  |                      6.5244e-05  | 2.97×         |
| Linear Octree, HilbertEncoder3D                         |                365.492 |                      3.65492e-05 | 5.31×         |
| Linear Octree, HilbertEncoder3D, Point + PointMetadata  |                372.619 |                      3.72619e-05 | 5.21×         |


### Benchmark Results for cloud: **5135_54435** at dataset: **Dales_LAS**
  - **Operation**: neighSearch
  - **Kernel type**: Sphere
  - **Dataset size**: 14196538
  - **Average points found per search**: 13407
  - **Repeats**: 3
  - **With warmup**: Yes

| Method                                                  |   Runtime (Total) (ms) |   Mean Runtime (Per Search) (ms) | Improvement   |
|:--------------------------------------------------------|-----------------------:|---------------------------------:|:--------------|
| Pointer Octree, Unencoded                               |                846.557 |                      5.96312e-05 | 1.00×         |
| Pointer Octree, MortonEncoder3D                         |                782.159 |                      5.50951e-05 | 1.08×         |
| Pointer Octree, HilbertEncoder3D                        |                660.821 |                      4.6548e-05  | 1.28×         |
| Pointer Octree, HilbertEncoder3D, Point + PointMetadata |                491.325 |                      3.46088e-05 | 1.72×         |
| Linear Octree, MortonEncoder3D                          |                665.62  |                      4.68861e-05 | 1.27×         |
| Linear Octree, HilbertEncoder3D                         |                491.647 |                      3.46315e-05 | 1.72×         |
| Linear Octree, HilbertEncoder3D, Point + PointMetadata  |                343.785 |                      2.42161e-05 | 2.46×         |


### Benchmark Results for cloud: **5135_54435** at dataset: **Dales_LAS**
  - **Operation**: numNeighSearch
  - **Kernel type**: Sphere
  - **Dataset size**: 14196538
  - **Average points found per search**: 13407
  - **Repeats**: 3
  - **With warmup**: Yes

| Method                                                  |   Runtime (Total) (ms) |   Mean Runtime (Per Search) (ms) | Improvement   |
|:--------------------------------------------------------|-----------------------:|---------------------------------:|:--------------|
| Pointer Octree, Unencoded                               |                714.78  |                      5.03489e-05 | 1.00×         |
| Pointer Octree, MortonEncoder3D                         |                782.55  |                      5.51226e-05 | 0.91×         |
| Pointer Octree, HilbertEncoder3D                        |                600.31  |                      4.22857e-05 | 1.19×         |
| Pointer Octree, HilbertEncoder3D, Point + PointMetadata |                460.807 |                      3.24591e-05 | 1.55×         |
| Linear Octree, MortonEncoder3D                          |                469.825 |                      3.30943e-05 | 1.52×         |
| Linear Octree, HilbertEncoder3D                         |                320.964 |                      2.26086e-05 | 2.23×         |
| Linear Octree, HilbertEncoder3D, Point + PointMetadata  |                242.434 |                      1.7077e-05  | 2.95×         |


### Benchmark Results for cloud: **sg27_station8_intensity_rgb** at dataset: **Semantic3D**
  - **Operation**: neighSearch
  - **Kernel type**: Sphere
  - **Dataset size**: 429615314
  - **Average points found per search**: 1396600
  - **Repeats**: 3
  - **With warmup**: Yes

| Method                                                  |   Runtime (Total) (ms) |   Mean Runtime (Per Search) (ms) | Improvement   |
|:--------------------------------------------------------|-----------------------:|---------------------------------:|:--------------|
| Pointer Octree, Unencoded                               |                61938   |                      0.000144171 | 1.00×         |
| Pointer Octree, MortonEncoder3D                         |                42207.1 |                      9.82439e-05 | 1.47×         |
| Pointer Octree, HilbertEncoder3D                        |                42210.6 |                      9.82521e-05 | 1.47×         |
| Pointer Octree, HilbertEncoder3D, Point + PointMetadata |                23984.6 |                      5.58281e-05 | 2.58×         |
| Linear Octree, MortonEncoder3D                          |                42842.2 |                      9.97222e-05 | 1.45×         |
| Linear Octree, HilbertEncoder3D                         |                35490.3 |                      8.26095e-05 | 1.75×         |
| Linear Octree, HilbertEncoder3D, Point + PointMetadata  |                23644.7 |                      5.50369e-05 | 2.62×         |


### Benchmark Results for cloud: **sg27_station8_intensity_rgb** at dataset: **Semantic3D**
  - **Operation**: numNeighSearch
  - **Kernel type**: Sphere
  - **Dataset size**: 429615314
  - **Average points found per search**: 1396600
  - **Repeats**: 3
  - **With warmup**: Yes

| Method                                                  |   Runtime (Total) (ms) |   Mean Runtime (Per Search) (ms) | Improvement   |
|:--------------------------------------------------------|-----------------------:|---------------------------------:|:--------------|
| Pointer Octree, Unencoded                               |               43555.3  |                      0.000101382 | 1.00×         |
| Pointer Octree, MortonEncoder3D                         |               37636.9  |                      8.7606e-05  | 1.16×         |
| Pointer Octree, HilbertEncoder3D                        |               36545.9  |                      8.50666e-05 | 1.19×         |
| Pointer Octree, HilbertEncoder3D, Point + PointMetadata |               20332    |                      4.73261e-05 | 2.14×         |
| Linear Octree, MortonEncoder3D                          |                9489.16 |                      2.20876e-05 | 4.59×         |
| Linear Octree, HilbertEncoder3D                         |                9376.58 |                      2.18255e-05 | 4.65×         |
| Linear Octree, HilbertEncoder3D, Point + PointMetadata  |                4802.57 |                      1.11788e-05 | 9.07×         |


### Benchmark Results for cloud: **Speulderbos_2017_TLS** at dataset: **Speulderbos**
  - **Operation**: neighSearch
  - **Kernel type**: Sphere
  - **Dataset size**: 721810646
  - **Average points found per search**: 61408
  - **Repeats**: 3
  - **With warmup**: Yes

| Method                                                  |   Runtime (Total) (ms) |   Mean Runtime (Per Search) (ms) | Improvement   |
|:--------------------------------------------------------|-----------------------:|---------------------------------:|:--------------|
| Pointer Octree, Unencoded                               |                2334.3  |                      3.23395e-06 | 1.00×         |
| Pointer Octree, MortonEncoder3D                         |                1892.63 |                      2.62206e-06 | 1.23×         |
| Pointer Octree, HilbertEncoder3D                        |                2009.08 |                      2.78339e-06 | 1.16×         |
| Pointer Octree, HilbertEncoder3D, Point + PointMetadata |                1112.94 |                      1.54187e-06 | 2.10×         |
| Linear Octree, MortonEncoder3D                          |                1604.07 |                      2.22229e-06 | 1.46×         |
| Linear Octree, HilbertEncoder3D                         |                1752.7  |                      2.4282e-06  | 1.33×         |
| Linear Octree, HilbertEncoder3D, Point + PointMetadata  |                 883.98 |                      1.22467e-06 | 2.64×         |


### Benchmark Results for cloud: **Speulderbos_2017_TLS** at dataset: **Speulderbos**
  - **Operation**: numNeighSearch
  - **Kernel type**: Sphere
  - **Dataset size**: 721810646
  - **Average points found per search**: 61408
  - **Repeats**: 3
  - **With warmup**: Yes

| Method                                                  |   Runtime (Total) (ms) |   Mean Runtime (Per Search) (ms) | Improvement   |
|:--------------------------------------------------------|-----------------------:|---------------------------------:|:--------------|
| Pointer Octree, Unencoded                               |               1941.92  |                      2.69035e-06 | 1.00×         |
| Pointer Octree, MortonEncoder3D                         |               1732.71  |                      2.4005e-06  | 1.12×         |
| Pointer Octree, HilbertEncoder3D                        |               1795.6   |                      2.48763e-06 | 1.08×         |
| Pointer Octree, HilbertEncoder3D, Point + PointMetadata |                918.404 |                      1.27236e-06 | 2.11×         |
| Linear Octree, MortonEncoder3D                          |                631.79  |                      8.75285e-07 | 3.07×         |
| Linear Octree, HilbertEncoder3D                         |                599.748 |                      8.30894e-07 | 3.24×         |
| Linear Octree, HilbertEncoder3D, Point + PointMetadata  |                333.678 |                      4.62279e-07 | 5.82×         |

