import resource, sys, numpy as np, os
CAP = 16 * 1024**3                      # 16 GB 하드 상한 (공유 장비 보호)
resource.setrlimit(resource.RLIMIT_AS, (CAP, CAP))
town = sys.argv[1]
p = f'/home/ailab/AILabDataset/01_Open_Dataset/41_Bench2Drive/Bench2Drive-Map/{town}_HD_map.npz'
print(f'{town}: {os.path.getsize(p)/1e9:.2f} GB 로딩 시도 (상한 16 GB)', flush=True)
try:
    m = dict(np.load(p, allow_pickle=True)['arr'])
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6
    npoly = sum(len(v) for k, v in m.items() for kk, v in [(kk, vv) for kk, vv in v.items()])
    print(f'  성공: road {len(m)}개, peak RSS {rss:.1f} GB')
except MemoryError:
    print(f'  MemoryError — 16 GB 로 부족')
except Exception as e:
    print(f'  실패: {type(e).__name__}: {e}')
