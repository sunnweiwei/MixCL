def do_multiprocessing(func, data, processes=20):
    import multiprocessing
    pool = multiprocessing.Pool(processes=processes)

    if isinstance(data, tuple):
        for item in data:
            assert len(item) == len(data[0])
        length = len(data[0]) // processes + 1
    else:
        length = len(data) // processes + 1
    results = []
    for ids in range(processes):
        if isinstance(data, tuple):
            collect = (item[ids * length:(ids + 1) * length] for item in data)
            results.append(pool.apply_async(func, (ids, *collect)))
        else:
            collect = data[ids * length:(ids + 1) * length]
            results.append(pool.apply_async(func, (ids, collect)))
    pool.close()
    pool.join()
    collect = []
    for j, res in enumerate(results):
        ids, result = res.get()
        assert j == ids
        collect.extend(result)
    return collect


def mp(func, data, processes=20, **kwargs):
    import multiprocessing
    pool = multiprocessing.Pool(processes=processes)
    length = len(data) // processes + 1
    results = []
    for ids in range(processes):
        collect = data[ids * length:(ids + 1) * length]
        results.append(pool.apply_async(func, args=(collect, ), kwds=kwargs))
    pool.close()
    pool.join()
    collect = []
    for j, res in enumerate(results):
        result = res.get()
        collect.extend(result)
    return collect
