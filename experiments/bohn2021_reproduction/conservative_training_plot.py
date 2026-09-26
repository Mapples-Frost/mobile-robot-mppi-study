"""All-scene/member fit-gap diagnostic; no independent-evaluation claims."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
from conservative_iteration import OUT
from paper_h_soft_probe import read, digest
from run import write


def prepare(round_id):
    source=OUT/'training_diagnosis/diagnosis.json';data=read(source)
    for p,h in data['hashes'].items():
        assert digest(Path(p))==h
    dest=OUT/'training_diagnosis'/('round%d_figures'%round_id);dest.mkdir(exist_ok=True)
    snapshots=OUT/'training_diagnosis/snapshots';snapshots.mkdir(exist_ok=True)
    source_snapshot=snapshots/(digest(source)+'.json')
    if source_snapshot.exists():assert source_snapshot.read_bytes()==source.read_bytes()
    else:source_snapshot.write_bytes(source.read_bytes())
    groups={}
    for r in data['members']:
        if r['round']!=round_id:continue
        key=(r['task'],r['seed'],r['member'],r['case'],r['out_of_bag'])
        groups.setdefault(key,[]).append(r)
    assert len(groups)==2*3*3*8
    rows=[]
    for (task,seed,member,case,oob),rr in sorted(groups.items()):
        rows.append(dict(task=task,seed=seed,member=member,case=case,
            membership='Out of bag' if oob else 'In bag',anchors=len(rr),
            transformed_mae=float(np.mean([r['transformed_mae'] for r in rr])),
            raw_argmin_regret=float(np.mean([r['raw_argmin_regret'] for r in rr])),
            safety_false_positives=sum(r['safety_false_positive'] for r in rr)))
    with (dest/'all_member_scenes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(dest/'input_audit.json',dict(rows=len(rows),round=round_id,source_hash=digest(source),
        csv_hash=digest(dest/'all_member_scenes.csv'),complete_all_tasks_seeds_members_scenes=True,
        question='Does low fitting error also hold on training scenes omitted by each bootstrap member?',
        scope='Training diagnosis; shared feature normalization and other members may have seen out-of-bag scenes. No independent samples or efficacy inference.',
        plot='Six panels, every member/scene dot. Log MAE axis; no deleted outliers, no confidence intervals. Internal report, general style.'))
    print(str(dest/'all_member_scenes.csv'))


def render(round_id,skill_scripts,export=False):
    sys.path.insert(0,str(skill_scripts))
    from visual_qa import audit_layout
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from PIL import Image
    dest=OUT/'training_diagnosis'/('round%d_figures'%round_id)
    assert digest(dest/'all_member_scenes.csv')==read(dest/'input_audit.json')['csv_hash']
    rows=list(csv.DictReader((dest/'all_member_scenes.csv').open()))
    values=np.array([float(r['transformed_mae']) for r in rows]);assert np.all(values>0) and np.isfinite(values).all()
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(3,2,figsize=(7.2,7.5),sharex=True,sharey=True)
    colors=['#0072B2','#D55E00'];markers=['o','s','^']
    for col,task in enumerate(('vehicle','pendulum')):
        for seed in range(3):
            ax=axes[seed,col]
            subset=[r for r in rows if r['task']==task and int(r['seed'])==seed]
            assert len(subset)==24
            for r in subset:
                member=int(r['member']);case=int(r['case']);x=int(r['membership']=='Out of bag')
                # Deterministic jitter preserves every case/member with no selection.
                jitter=((member*8+case)/23-.5)*.38
                ax.scatter(x+jitter,float(r['transformed_mae']),color=colors[x],marker=markers[member],
                           s=24,alpha=.85,edgecolors='black',linewidths=.25)
            ax.set_yscale('log');ax.set_ylim(values.min()/2,values.max()*2)
            ax.set_xlim(-.4,1.4);ax.set_xticks([0,1]);ax.set_xticklabels(['In bag','Out of bag'])
            ax.set_title('%s, seed %d'%(task.capitalize(),seed),fontsize=10)
            ax.grid(axis='y',alpha=.2)
            if col==0:ax.set_ylabel('Mean absolute label error\n(signed-log cost units)')
    fig.suptitle('Round %d training fit: all scenes and ensemble members'%round_id,y=.985,fontsize=11)
    fig.legend(handles=[Line2D([],[],marker=m,color='black',linestyle='None',label='Member %d'%i) for i,m in enumerate(markers)],
               loc='lower center',bbox_to_anchor=(.5,.055),ncol=3,frameon=False)
    fig.text(.5,.015,'Each dot: one member on one training scene, averaged over its anchors.\nShared training normalization; this is not independent validation.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.115,1,.96))
    fig.savefig(dest/'fit_gap_preview.png',dpi=180)
    Image.open(dest/'fit_gap_preview.png').convert('L').save(dest/'fit_gap_grayscale.png')
    issues=audit_layout(fig)
    write(dest/'layout_audit.json',dict(issues=issues,rows_plotted=len(rows),minimum_value=float(values.min()),
        maximum_value=float(values.max()),source_hash=digest(Path(__file__))))
    if export:
        for suffix in ('pdf','svg'):fig.savefig(dest/('fit_gap.'+suffix))
        fig.savefig(dest/'fit_gap.png',dpi=300)
    plt.close(fig)
    print(json.dumps(dict(rows=len(rows),issues=issues)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--round',type=int,default=0)
    ap.add_argument('--mode',choices=['prepare','render'],required=True)
    ap.add_argument('--skill-scripts',type=Path)
    ap.add_argument('--export',action='store_true');a=ap.parse_args()
    if a.mode=='prepare':prepare(a.round)
    else:render(a.round,a.skill_scripts,a.export)
