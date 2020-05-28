from tqdm import tqdm
from multiprocessing import Pool


def pool_map(fn, obj_list, collect=False):
    pbar = tqdm(total=len(obj_list))
    res = []

    def collect_result(result):
        pbar.update()

        if collect:
            res.append(result)

    pool = Pool()
    for i in range(pbar.total):
        pool.apply_async(fn, args=(obj_list[i], ), callback=collect_result) # obj_list[i] might be tuple?
    pool.close()
    pool.join()

    return res
