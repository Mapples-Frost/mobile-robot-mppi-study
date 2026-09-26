"""All pilot branch curves, including unsafe candidates and negative results."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'research_artifacts/bohn2021_reproduction_2026-09-17/results/fixed_policy_branches_2026-09-24'
rows=list(csv.DictReader((OUT/'all_branches.csv').open()))
audit=json.loads((OUT/'mechanism_audit.json').read_text())
assert audit['passed']
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'savefig.dpi':180})
fig,axes=plt.subplots(3,2,figsize=(10,9),sharex=True)
colors=['#0072B2','#D55E00'];markers={0:'o',25:'s',60:'^'}
for ti,task in enumerate(['vehicle','pendulum']):
    for seed in range(3):
        ax=axes[seed,ti]
        relevant=[r for r in rows if r['task']==task and int(r['seed'])==seed]
        pairs=sorted(set((int(r['case']),int(r['anchor'])) for r in relevant))
        for case,anchor in pairs:
            rr=sorted([r for r in relevant if int(r['case'])==case and int(r['anchor'])==anchor],key=lambda r:int(r['h']))
            ax.plot([int(r['h']) for r in rr],[float(r['cost_difference']) for r in rr],
                color=colors[case],marker=markers[anchor],markersize=4,linewidth=.9,alpha=.7)
            unsafe=[r for r in rr if r['admissible']=='False']
            ax.scatter([int(r['h']) for r in unsafe],[float(r['cost_difference']) for r in unsafe],
                       marker='x',s=48,color='black',linewidths=1.3,zorder=5)
        ax.axhline(0,color='gray',linewidth=.8)
        ax.set_yscale('symlog',linthresh=.1)
        values=[float(r['cost_difference']) for r in relevant]+[0.]
        transform=ax.yaxis.get_transform()
        lo,hi=transform.transform(np.array([min(values),max(values)]))
        pad=max((hi-lo)*.1,.015)
        ax.set_ylim(transform.inverted().transform(np.array([lo-pad,hi+pad])))
        s=next(s for s in audit['summaries'] if s['task']==task and s['seed']==seed)
        ax.set_title('%s, seed %d | %d anchors, %d skipped'%(task.capitalize(),seed,s['usable_anchors'],s['skipped_anchors']))
        ax.set_xticks([1,10,20,30,40,50]);ax.grid(axis='y',alpha=.18)
        ax.set_ylabel('Cost difference vs fixed-H suffix\n(symlog; lower is better)')
        if seed==2:ax.set_xlabel('Forced first-step prediction horizon H')
legend=[Line2D([0],[0],color=c,label='Training case %d'%i) for i,c in enumerate(colors)]
legend += [Line2D([0],[0],color='gray',marker=m,linestyle='None',label='Anchor %d'%a) for a,m in markers.items()]
legend += [Line2D([0],[0],color='black',marker='x',linestyle='None',label='Safety criterion fails')]
fig.suptitle('Fixed-policy branch pilot: all seeds and candidates (training only)',y=.995)
fig.legend(handles=legend,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.005))
fig.tight_layout(rect=[0,.075,1,.975])
for suffix in ('png','pdf'):fig.savefig(OUT/('all_branch_differences.'+suffix))
plt.close(fig)
