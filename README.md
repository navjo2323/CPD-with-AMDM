# Alternating Mahalanobis Distance Minimization for Accurate and Well-Conditioned CP Decomposition

This repository contains the data and code used in Alternating Mahalanobis Distance Minimization for Accurate and Well-Conditioned CP Decomposition [link](https://arxiv.org/abs/2204.07208). The code used to generate all the figures and Algorithm 5.1 is
[here](https://github.com/cyclops-community/tensor_decomposition/blob/master/mahalanobis.py)

For AMDM and hybrid algorithms the following file should be run in the above mentioned repository
```
python mahalanobis.py
```
with parameters as described below.

```
python mahalanobis.py --tlib numpy --order o --tensor ten --col (c1,c2) --R r --R-app ra --num-iter n 
--fit f --thresh t --reduce-thresh rt --reduce-thresh-freq rtf 
--tol to --compute-cond cc 
```
where 
- `o` is the order of the input tensor
- `ten` is the input tensor name, it is `random` for Random tensor, `random-col` for collinearity tensor, `amino` for amino acid tensor, `SLEEP` for the SLEEP-EDF tensor, `MGH` for MGH tensor, `scf` for the SCF tensor
- `col` controls the collinearity of factors which lies between $(c_1,c_2)$, for setting a collinearity to be some value c, c1=c2= c should be used
- `r` is the rank of the synthetic tensor (valid for `random` and `random-col` tensors)
- `ra` is the rank used to approximate the input tensor
- `n` is the number of total iterations
- `f` is the fitness tolerance to terminate the algorithm
- `t` is the threshhold for inverting components in the hybrid algorithm as described in Algorithm 5.1 in the paper
- `rt` is used to vary the threshhold. Should be 1 when we need to reduce the threshold after every certain number of iterations, 0 otherwise
- `rtf` is used to control the number of iterations after which the threshold should be reduced
- `cc` is used if the condition number of CPD is computed in every iteration. The computation follows the efficient implementation as described in Appendix A


The Code folder contains AMDM code for Algorithm 3.1 and ALS code used to generate data for Figure 3 in the paper.

The Data folder contains all the data used to generate all the other figures in the paper. Generating_plots.ipynb is a notebook that has code for producing each figure in the paper.
