from tqdm import tqdm
from multiprocessing import Pool


def pool_map(fn, obj_list):
    pbar = tqdm(total=len(obj_list))

    def collect_result(result):
        pbar.update()

    pool = Pool()
    for i in range(pbar.total):
        pool.apply_async(fn, args=(obj_list[i], ), callback=collect_result)
    pool.close()
    pool.join()

    return
