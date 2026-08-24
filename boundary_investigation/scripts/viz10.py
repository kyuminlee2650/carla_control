import numpy as np, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

ORDER=['24367','27582','3144','2416','2715','2286','17569','2790','3373','1792']
M={r['route']:r for r in json.load(open('metrics.json'))}
data={t:np.load(f'carla_{t}.npy',allow_pickle=True).item() for t in ['Town06','Town11','Town12','Town13']}
loc={rid:t for t in data for rid in data[t]['corridor']}

fig,axes=plt.subplots(2,5,figsize=(30,13))
for ax,rid in zip(axes.ravel(),ORDER):
    t=loc[rid]; d=data[t]; info=d['corridor'][rid]; snap=info['snap']; m=M[rid]
    C=snap[:,:2]; cx,cy=C.mean(0); half=max(C.max(0)-C.min(0)).max()/2+55

    ax.add_collection(LineCollection(d['bands'],colors='#dcdcdc',linewidths=2.0,zorder=0))
    for ty,P,rr,ll,isj,side in d['polys']:
        if ty=='Boundary': continue
        if ty=='Center': ax.plot(P[:,0],P[:,1],color='#c4c4c4',lw=0.5,ls=(0,(4,4)),zorder=1)
        elif ty=='SolidSolid': ax.plot(P[:,0],P[:,1],color='#f5a623',lw=2.0,zorder=2)
        else: ax.plot(P[:,0],P[:,1],color='#4a90d9',lw=0.9,zorder=2)
    for ty,P,*_ in d['polys']:
        if ty=='Boundary': ax.plot(P[:,0],P[:,1],color='#e02020',lw=2.6,zorder=4)
    ax.plot(C[:,0],C[:,1],color='#111111',lw=2.0,zorder=5)
    ax.plot(C[0,0],C[0,1],'o',color='#111',ms=7,zorder=6)

    # ego box 하나 예시 (경로 중간 지점)
    i=len(snap)//2; x,y,yaw,_=snap[i]; a=np.deg2rad(yaw)
    f=np.array([np.cos(a),np.sin(a)]); r=np.array([-np.sin(a),np.cos(a)])
    box=np.array([[-15,-30],[15,-30],[15,30],[-15,30],[-15,-30]],float)
    W=np.array([x,y])+box[:,0:1]*r+box[:,1:2]*f
    ax.plot(W[:,0],W[:,1],color='#2e7d32',lw=1.6,ls='--',zorder=6)

    ax.set_xlim(cx-half,cx+half); ax.set_ylim(cy+half,cy-half); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"route {rid}  ({t})\n{m['scenario']}",fontsize=10.5)
    ax.text(0.02,0.975,f"SolidSolid (current): {m['ss_pct']:.0f}% of frames, {m['ss_mean']:.2f} inst\n"
                       f"Boundary  (fixed)  : {m['bd_pct']:.0f}% of frames, {m['bd_mean']:.2f} inst",
            transform=ax.transAxes,va='top',ha='left',fontsize=8.6,family='monospace',
            bbox=dict(fc='white',ec='#999',alpha=.92,pad=3.5))

h=[plt.Line2D([],[],color=c,lw=w,ls=s) for c,w,s in
   [('#e02020',2.6,'-'),('#f5a623',2.0,'-'),('#4a90d9',1.2,'-'),('#c4c4c4',.9,(0,(4,4))),
    ('#111111',2.0,'-'),('#2e7d32',1.6,'--')]]
fig.legend(h,['Boundary (auto-labelled)','SolidSolid (what the loss uses today)','other lane markings',
              'lane centre','planned route','ego BEV box 30x60 m'],
           loc='lower center',ncol=6,frameon=False,fontsize=11.5)
fig.suptitle('Bench2Drive 10-route set — road boundary labelling (RED) vs the current SolidSolid proxy (ORANGE)',
             fontsize=15,y=0.985)
plt.tight_layout(rect=[0,0.035,1,0.965])
plt.savefig('b2d10_boundary.png',dpi=100,bbox_inches='tight')
print('saved')
