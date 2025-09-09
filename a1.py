#!/usr/bin/env python3

import numpy as np
import requests as rq
import json
from scipy.optimize import linprog, lsq_linear

def post(challenge_id, vector, submit):
    assert challenge_id.isalnum()
    assert np.all(np.abs(vector) == 1)

    url = "https://rasmuspagh.pythonanywhere.com/query"
    payload = {
        "challengeid": challenge_id,
        "submit": submit,
        "query": str(np.atleast_2d(vector).tolist()),
    }
    response = rq.post(url, data=payload).json()
    return response["result"]

def query(challenge_id, query_vector):
    """Retrieve answer to challenge for a given query"""
    return json.loads(post(challenge_id, query_vector, submit=False))

def submit(challenge_id, dataset_vector):
    """Submit a candidate dataset to challenge"""
    return post(challenge_id, dataset_vector, submit=True)


challenge_id = "Identifier1Secret"
n_entries = 256
n_queries = 512

# Re-use queries during development
try:
    query_results, queries
except NameError:
    queries = np.random.choice([-1, +1], size=(n_queries, n_entries))
    query_results = np.array(query(challenge_id, queries))

# Reconstruction using linprog
noise = 38
c = np.ones(512)
A_ub = np.block([
    [+queries, -queries],
    [-queries, +queries]
])
b_ub = np.block([
    noise+query_results,
    noise-query_results
])
res = linprog(c, A_ub, b_ub, bounds = (0,1))
r = ((res.x[256:]-res.x[:256]) < 0).astype(np.int64)*2-1

# Reconstruction using lsq_linear
res2 = lsq_linear(queries, query_results, bounds=(-1, 1))
r2 = (res2.x > 0).astype(np.int64)*2-1

best_query_result = submit(challenge_id, r2)
print(
    "\nReconstruction attack achieves fraction "
    f"{(1 + best_query_result / n_entries) / 2} correct values"
)
