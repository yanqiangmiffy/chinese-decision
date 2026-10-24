"""Cutting encoded items into batches by row count and token budget."""
import random


def greedy_batches(items, batch_size, token_budget):
    """Fill length-sorted items in order until the row or (unpadded) token limit is reached."""
    batches, batch = [], []
    for item in items:
        if batch and (len(batch) >= batch_size or (len(batch) + 1) * len(item["input_ids"]) > token_budget):
            batches.append(batch)
            batch = []
        batch.append(item)
    if batch:
        batches.append(batch)
    return batches


def shape_batches(items, batch_size, token_budget):
    """Cut length-sorted items into power-of-two batches whose 64-padded size fits the budget.

    Few distinct (rows, padded length) shapes keep compiled kernels and memory stable.
    """
    batches, position = [], 0
    while position < len(items):
        capacity = min(batch_size, len(items) - position)
        size = 1 << (capacity.bit_length() - 1)
        while size > 1:
            padded = ((len(items[position + size - 1]["input_ids"]) + 63) // 64) * 64
            if size * padded <= token_budget:
                break
            size //= 2
        batches.append(items[position:position + size])
        position += size
    return batches


def token_batches(stream, processor, batch_size, token_budget, seed, stats, pool_size=128):
    """Encode a stream of examples in pools, cut each pool into shape batches, and shuffle them.

    One seeded generator drives both candidate shuffling and batch order, so a resumed
    run can replay the stream exactly up to its saved batch cursor.
    """
    rng, pool = random.Random(seed), []

    def flush(items):
        items.sort(key=lambda e: len(e["input_ids"]))
        batches = shape_batches(items, batch_size, token_budget)
        rng.shuffle(batches)
        return batches
    for example in stream:
        item = processor.encode_example(example, rng, stats)
        if item is None:
            continue
        pool.append(item)
        if len(pool) >= pool_size:
            yield from flush(pool)
            pool = []
    if pool:
        yield from flush(pool)
