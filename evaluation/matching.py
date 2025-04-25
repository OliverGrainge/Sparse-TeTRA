import faiss
import numpy as np
import torch


def match_cosine_gpu(
    global_desc, num_references, ground_truth, k_values=[1, 5, 10], gpu_id=0
):
    if global_desc.dtype == torch.float16:
        global_desc = global_desc.float()

    global_desc = torch.nn.functional.normalize(
        global_desc, dim=1
    )  # L2-normalize for cosine
    global_desc = (
        global_desc.detach().cpu().numpy().astype(np.float32)
    )  # FAISS needs float32 on CPU first

    reference_desc = global_desc[:num_references]
    query_desc = global_desc[num_references:]

    dim = reference_desc.shape[1]

    # Initialize GPU resources
    res = faiss.StandardGpuResources()
    index_cpu = faiss.IndexFlatIP(
        dim
    )  # inner product = cosine similarity after normalization
    index = faiss.index_cpu_to_gpu(res, gpu_id, index_cpu)

    index.add(reference_desc)
    dist, predictions = index.search(query_desc, max(k_values))

    correct_at_k = np.zeros(len(k_values))
    for q_idx, pred in enumerate(predictions):
        if not isinstance(ground_truth[q_idx], (list, set, tuple, np.ndarray)):
            gt_set = {ground_truth[q_idx]}
        else:
            gt_set = set(ground_truth[q_idx])
        for i, k in enumerate(k_values):
            if any(p in gt_set for p in pred[:k]):
                correct_at_k[i:] += 1
                break

    correct_at_k = (correct_at_k / len(predictions)) * 100
    return {k: float(v) for k, v in zip(k_values, correct_at_k)}


def match_cosine_cpu(global_desc, num_references, ground_truth, k_values=[1, 5, 10]):
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


def match_cosine(
    global_desc, num_references, ground_truth, k_values=[1, 5, 10], use_gpu=False
):
    if use_gpu == False:
        return match_cosine_cpu(global_desc, num_references, ground_truth, k_values)
    else:
        return match_cosine_gpu(global_desc, num_references, ground_truth, k_values)
