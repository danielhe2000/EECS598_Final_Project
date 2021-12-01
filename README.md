# Cryptohash Recovery

## Baseline
Provided CPU baseline is located in [baseline](baseline), while our GPU baseline is located in [cuda_v1](cuda_v1). To run the baselines,
1. Build the executable code:
`make`

2. Run the cryptohash kernel:
`sbatch run_test [32-character hash]`

You can find the 32-bit character of your desired string by:

1. Build the executable code:
`make wiki`

2. Run the wiki kernel:
`./wiki [string]`

## Optimizations

The three optimizations mentioned in the report are located in [cuda_v2](cuda_v2), [cuda_v3](cuda_v3), and [cuda_v4](cuda_v4) respectively. [cuda_v5](cuda_v5) and [cuda_v6](cuda_v6) are two experimental optimizations that are not eventually adopted in the report. The process of running the optimized code is the same as running the baseline. 

## Testing
[generate_testcase](generate_testcase) is used to generate the test sets, which are stored in [Testcase](Testcase). All folders ending in "\_test" are used for testing. To run the testing codes, 
1. Build the executable code:
`make`

2. Run the cryptohash kernel:
`sbatch run_test [number of passwords to be tested]`

