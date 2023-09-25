import dask
from dask_cuda.worker_spec import *
from dask.distributed import Client, Scheduler, Worker, Nanny, SpecCluster
import multiprocessing
import psutil
import os
import imagej
import imagej.doctor

def init_dask_cluster(cpu_scale=3, gpu_scale=1):
    print("Initializing Dask cluster...")

    # https://github.com/rapidsai/dask-cuda/issues/1161
    # gather device info
    cpu_count = multiprocessing.cpu_count()
    memory_count = psutil.virtual_memory().total
    print("CPU count:", cpu_count)
    print("System memory:",memory_count)

    specs = {
        "cpu":{
            "scale":cpu_scale,
            "resources":{
            }
        },
        "gpu":{
            "scale":gpu_scale,
            "resources":{
                "CUDA_VISIBLE_DEVICES": [0]
            }
        }
    }

    worker_count = 0
    for v in specs.values():
        worker_count += v["scale"]

    nthreads = cpu_count//worker_count
    memory_limit = int(memory_count*0.9)//worker_count # set to use 90% of the system memory to avoid crashing

    print("number of workers:", worker_count)
    print("threads per worker:", nthreads)
    print("memory limit per worker:", round(memory_limit/(1024*1024*1024),2), "GB")

    workers = {}

    for k, v in specs.items():
        for i in range(v["scale"]):
            print(v)
            if "CUDA_VISIBLE_DEVICES" in v["resources"].keys():
                workers["{}-{}".format(k,i)] = worker_spec(threads_per_worker=nthreads, memory_limit=memory_limit, CUDA_VISIBLE_DEVICES=v["resources"]["CUDA_VISIBLE_DEVICES"])[0]
            else:
                workers["{}-{}".format(k,i)] = {
                    "cls":Nanny,
                    "options":{
                        "nthreads": nthreads,
                        "memory_limit": memory_limit
                        }
                }     

    scheduler = {'cls': Scheduler, 'options': {"dashboard_address": ':8787'}}
    cluster = SpecCluster(scheduler=scheduler, workers=workers)
    client = Client(cluster)
    
    return client

def pyimagej_init(FIJI_DIR=""):
    imagej.doctor.checkup()

    if not os.path.exists(FIJI_DIR) or FIJI_DIR == "":
        raise Exception("Fiji.app directory not found")

    print("Initializing Fiji on JVM...")
    ij = imagej.init(FIJI_DIR,mode='headless')
    print(ij.getApp().getInfo(True))