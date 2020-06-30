# About Lab Matrix Multiplication

Visual Studio 2019 and Cuda Toolkit 10 were used for this work.

## System configuration

| Name  | Values  |
|-------|---------|
| CPU  | AMD Ryzen 5 3600X, BOX |
| RAM  | 8 GB DDR4 |
| GPU  | Gigabyte GeForce GTX 1050 Ti|
| OS   | Windows 10 64-bit  |

## Results

The average time in milliseconds for 5 measurements.

|    Size     |          CPU        |         GPU       |
|-------------|---------------------|-------------------|
| 256 х 256   | 63 ms               | 3.2 ms            |
| 512 х 512   | 450 ms              | 24 ms             |
| 1024 х 1024 | 8245 ms             | 154 ms            |
| 2048 х 2048 | 20240 ms            | 1576 ms           |
