import random
from typing import List

import numpy as np
from sklearn.linear_model import LogisticRegression

from data import (
    collect_newsgroup_data,
    collect_review_data,
    collect_spam_a_data,
    collect_spam_b_data,
)
from example import Example
from lwe import LWE, SMA, labels, pLWE, score

# # reviews
# b_data, d_data, e_data, k_data = collect_review_data(5000)
# print("data parsed")
# N = 1000
# for i in range(3):
#     train = [random.sample(d_data, N), random.sample(b_data, N)]
#     models = [LogisticRegression(solver="liblinear") for data in train]
#     test = random.sample(k_data, 500)

#     for i, model, data in zip(range(len(models)), models, train):
#         model.fit(data, labels(data))
#         print(f"Model {i} fitted")

#     print(f"Baseline Accuracy: {score(SMA(train, models, test, 0.5, 2), labels(test))}")
#     print(f"LWE Accuracy: {score(LWE(train, models, test, 0.5, 2), labels(test))}")
#     print(f"pLWE Accuracy: {score(pLWE(train, models, test, 0.5, 2), labels(test))}")


# # Spam Task A Data
# data = collect_spam_a_data(5000)
# for i in range(3):
#     source = data[0]
#     target = data[1]
#     print("data parsed")
#     train = [source]
#     models = [LogisticRegression(solver="liblinear") for data in train]
#     test = random.sample(target, len(target) // 2)

#     for i, model, dataset in zip(range(len(models)), models, train):
#         model.fit(dataset, labels(dataset))
#         print(f"Model {i} fitted")
#     print(f"Baseline Accuracy: {score(SMA(train, models, test, 0.5, 2), labels(test))}")
#     print(f"LWE Accuracy: {score(LWE(train, models, test, 0.5, 2), labels(test))}")
#     print(f"pLWE Accuracy: {score(pLWE(train, models, test, 0.5, 2), labels(test))}")
## Spam Task B Data
# data = collect_spam_b_data(3000)
# source1 = data[0]
# source2 = data[1]
# target = data[2]
# print("data parsed")
# train = [source1, source2]
# models = [LogisticRegression(solver="liblinear") for data in train]
# test = random.sample(target, len(target))
# for i, model, dataset in zip(range(len(models)), models, train):
#     model.fit(dataset, labels(dataset))
#     print(f"Model {i} fitted")
# print(f"Baseline: {score(SMA(train, models, test, 0.5, 2), labels(test))}")
# print(f"LWE Accuracy: {score(LWE(train, models, test, 0.5, 2), labels(test))}")
# print(f"pLWE Accuracy: {score(pLWE(train, models, test, 0.5, 2), labels(test))}")

# newsgroup
# nws1, nws2 = collect_newsgroup_data(3000)

# print("data parsed")
# train = [nws1]
# models = [LogisticRegression(solver="liblinear") for data in train]
# test = random.sample(nws2, len(nws2))
# for i, model, dataset in zip(range(len(models)), models, train):
#     model.fit(dataset, labels(dataset))
#     print(f"Model {i} fitted")
# print(f"Baseline: {score(SMA(train, models, test, 0.5, 2), labels(test))}")
# print(f"LWE Accuracy: {score(LWE(train, models, test, 0.5, 2), labels(test))}")
# print(f"pLWE Accuracy: {score(pLWE(train, models, test, 0.5, 2), labels(test))}")
