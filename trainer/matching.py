import torch 
import faiss 
import numpy as np 

def match_cosine(global_desc, num_references, ground_truth, k_values=[1, 5, 10]):
    if global_desc.dtype == torch.float16:
        global_desc = global_desc.float()
    global_desc = global_desc.cpu().numpy()
    reference_desc = global_desc[:num_references]
    query_desc = global_desc[num_references:]

    index = faiss.IndexFlatIP(reference_desc.shape[1])
    index.add(reference_desc)

    dist, predictions = index.search(query_desc, max(k_values))

    correct_at_k = np.zeros(len(k_values))
    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(k_values):
            if np.any(np.in1d(pred[:n], ground_truth[q_idx])):
                correct_at_k[i:] += 1
                break
    d = {}
    correct_at_k = (correct_at_k / len(predictions)) * 100
    for k, v in zip(k_values, correct_at_k):
        d[k] = v
    return d