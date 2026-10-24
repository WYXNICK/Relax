"""Deploy the repository retriever on one 80GB GPU with bounded clone peaks.

Four contiguous Flat shards live on the SAME GPU. IndexShards merges exact
top-k and restores original document IDs. Embeddings, FP16 storage, corpus,
HTTP protocol, batching and scoring remain the repository implementation.
"""

import gc
import sys
from functools import partial
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from examples.search_r1 import retrieval_server as server


def create_retriever(*, index_path, corpus_path, model_path, topk):
    cpu = server.faiss.read_index(index_path)
    if not isinstance(cpu, server.faiss.IndexFlat):
        raise TypeError("This deployment expects the fixed Wiki18 Flat index")
    retriever = server.E5FlatRetriever.__new__(server.E5FlatRetriever)
    retriever.index = server.faiss.IndexShards(cpu.d, False, True)
    retriever._gpu_resources = []
    retriever._gpu_shards = []
    block = (cpu.ntotal + 3) // 4
    for start in range(0, cpu.ntotal, block):
        count = min(block, cpu.ntotal - start)
        part = server.faiss.IndexFlat(cpu.d, cpu.metric_type)
        part.add(cpu.reconstruct_n(start, count))
        resource = server.faiss.StandardGpuResources()
        resource.setTempMemory(256 * 1024 * 1024)
        options = server.faiss.GpuClonerOptions()
        options.useFloat16 = True
        gpu = server.faiss.index_cpu_to_gpu(resource, 0, part, options)
        retriever.index.add_shard(gpu)
        retriever._gpu_resources.append(resource)
        retriever._gpu_shards.append(gpu)
        del part
        gc.collect()
        print(f"Loaded {start + count}/{cpu.ntotal} vectors on GPU 0", flush=True)
    assert retriever.index.ntotal == cpu.ntotal
    del cpu
    gc.collect()
    retriever.corpus = server.datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)
    assert len(retriever.corpus) == retriever.index.ntotal
    retriever.encoder = server.E5Encoder(model_path)
    retriever.topk = topk
    return retriever


if __name__ == "__main__":
    args = server.parse_args()
    server.app.state.create_retriever = partial(
        create_retriever,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        model_path=args.retriever_model,
        topk=args.topk,
    )
    for name in ("batch_wait_ms", "max_batch_queries", "max_pending_requests"):
        setattr(server.app.state, name, getattr(args, name))
    server.uvicorn.run(server.app, host="::", port=args.port)
