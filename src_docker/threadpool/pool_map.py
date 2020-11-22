from tqdm import tqdm
from multiprocessing import Pool

def pool_map(fn, obj_list, collect=False, arg_tuple=False):
    pbar = tqdm(total=len(obj_list))
    res = []

    def collect_result(result):
        pbar.update()

        if collect:
            res.append(result)

    pool = Pool()
    for i in range(pbar.total):
        # obj_list[i] might be tuple?
        if arg_tuple:
            pool.apply_async(
                fn, args=(*obj_list[i], ), callback=collect_result)
        else:
            pool.apply_async(fn, args=(obj_list[i], ), callback=collect_result)
    pool.close()
    pool.join()

    return res
