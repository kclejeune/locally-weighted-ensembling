# Overview: Locally Weighted Ensembling

Locally Weighted Ensembling is a transductive parameter transfer learning framework
intended to improve the learning of a target task T_T on a testing domain D_T, transferring 
knowledge from k models trained on k labeled domains of interest. For any example `x`, we 
can weight the model predictions according to their performance in the neighborhood 
of other examples clustered near `x`, making an overall prediction by constructing an ensemble which
is weighted according to structural similarity near `x` in the test domain.

## Relevant Research

This repository implements and analyzes results proposed in ![Gao et. al](papers/knowledge-transfer-lwe.pdf). 
