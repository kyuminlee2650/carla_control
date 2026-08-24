import xml.etree.ElementTree as ET, collections, statistics as st
root = ET.parse("/home/ailab/2026intern/carla/CarlaUE4/Content/Carla/Maps/OpenDrive/Town03.xodr").getroot()

norm, conn = [], []
for r in root.findall('road'):
    (conn if r.get('junction') != '-1' else norm).append(r)
print(f"Town03 .xodr:  <road> 총 {len(norm)+len(conn)}개   일반 {len(norm)}   교차로 연결로 {len(conn)}")
print(f"               <junction> 엘리먼트 {len(root.findall('junction'))}개")

def stats(rs, name):
    L   = [float(r.get('length')) for r in rs]
    nls = [len(r.findall('lanes/laneSection')) for r in rs]
    dl, both, mk = [], 0, collections.Counter()
    for r in rs:
        s0 = r.find('lanes/laneSection')
        ids = [int(l.get('id')) for side in ('left','right') for l in s0.findall(f'{side}/lane')]
        d   = [int(l.get('id')) for side in ('left','right') for l in s0.findall(f'{side}/lane') if l.get('type')=='driving']
        dl.append(len(d))
        if any(i<0 for i in d) and any(i>0 for i in d): both += 1
        for l in s0.findall('.//lane'):
            for rm in l.findall('roadMark'): mk[rm.get('type')] += 1
    print(f"\n--- {name} ({len(rs)}개) ---")
    print(f"  길이        중앙값 {st.median(L):7.1f} m   min {min(L):6.2f}   max {max(L):7.1f}")
    print(f"  laneSection 중앙값 {st.median(nls):.0f}개")
    print(f"  driving 차선 수 분포 {dict(sorted(collections.Counter(dl).items()))}")
    print(f"  양방향(±둘 다 driving) road: {both}/{len(rs)}")
    print(f"  roadMark 타입: {dict(mk.most_common(6))}")

stats(norm, "일반 구간  junction=\"-1\"")
stats(conn, "교차로 연결로  junction=\"<id>\"")

print("\n--- 연결로 predecessor/successor 는 무엇을 가리키나 ---")
c = collections.Counter()
for r in conn:
    for tag in ('predecessor','successor'):
        e = r.find(f'link/{tag}')
        c[(tag, e.get('elementType') if e is not None else None)] += 1
print("  연결로:", dict(c))
c = collections.Counter()
for r in norm:
    for tag in ('predecessor','successor'):
        e = r.find(f'link/{tag}')
        c[(tag, e.get('elementType') if e is not None else None)] += 1
print("  일반  :", dict(c))

j = root.findall('junction')[0]
print(f"\n--- junction id={j.get('id')} 의 connection 목록 (앞 8개) ---")
for cn in j.findall('connection')[:8]:
    ll = cn.find('laneLink')
    print(f"    들어오는 road {cn.get('incomingRoad'):>4}  ->  연결로 road {cn.get('connectingRoad'):>4}"
          f"   laneLink from={ll.get('from')} to={ll.get('to')}" if ll is not None else "")
print(f"    (이 junction 의 connection 총 {len(j.findall('connection'))}개)")
