"""All-seed validation cost/latency figure; duplicate comparators are not samples."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
from conservative_iteration import OUT,TASKS
from paper_h_soft_probe import read,digest
from run import write

DEST=OUT/'validation_figures'


def prepare():
    report=OUT/'all_candidate_delivery_timed/report.json';data=read(report)
    assert data['timing_available'] and not data['selection_eligible']
    for p,h in data['hashes'].items():assert digest(Path(p))==h
    rows=[]
    for task in TASKS:
        for r in range(2):
            for seed in range(3):
                a=next(v for v in data['comparisons'] if (v['task'],v['round'],v['seed'],v['comparator'])==(task,r,seed,'independent'))
                b=next(v for v in data['comparisons'] if (v['task'],v['round'],v['seed'],v['comparator'])==(task,r,seed,'matched'))
                assert {k:v for k,v in a.items() if k!='comparator'}=={k:v for k,v in b.items() if k!='comparator'}
                scale=max(abs(a['fixed_total_cost']),1e-12)
                rows.append(dict(task=task,round=r,seed=seed,fixed_h=a['fixed_h'],
                    cost_change_percent=100*a['relative_cost_difference'],
                    cost_lower_percent=100*a['cost_difference_lower']/scale,cost_upper_percent=100*a['cost_difference_upper']/scale,
                    time_change_percent=100*(a['timing_ratio']-1),
                    time_lower_percent=100*(a['timing_ratio_lower']-1),time_upper_percent=100*(a['timing_ratio_upper']-1),
                    time_repeat0_percent=100*(a['timing_repeat0_ratio']-1),time_repeat1_percent=100*(a['timing_repeat1_ratio']-1),
                    adaptive_success=a['adaptive_successes'],fixed_success=a['fixed_successes'],
                    adaptive_constraints=a['adaptive_constraint_episodes'],fixed_constraints=a['fixed_constraint_episodes'],
                    adaptive_failed_steps=a['adaptive_solver_failure_steps'],fixed_failed_steps=a['fixed_solver_failure_steps']))
    DEST.mkdir(exist_ok=True)
    with (DEST/'all_seed_comparisons.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write(DEST/'input_audit.json',dict(source=str(report),source_hash=digest(report),csv_hash=digest(DEST/'all_seed_comparisons.csv'),
        rows=12,all_seeds_rounds=True,duplicate_identical_comparator_labels_removed=True,
        question='Do either round and all three seeds simultaneously reduce cost or decision latency?',
        interval='95% percentile paired-scene bootstrap,24 shared scenes,10000 resamples, conditional on trained models and two observed timing repetitions; not adjusted for multiple comparisons.',
        limitations='Validation only; no independent test, no seed-population inference. Timing repeat markers reveal scheduling variation absent from conditional scene intervals. All rounds failed registered entry gates.',
        intended_plot='2 tasks x(cost,time) forest panels,6 model points each; include both timing repeat points. Symmetric-log cost axis only if required by observed range; all extremes retained. General internal-report style.'))
    print(str(DEST/'all_seed_comparisons.csv'))


def render(skill_scripts,export=False):
    sys.path.insert(0,str(skill_scripts))
    from visual_qa import audit_layout
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from PIL import Image
    assert digest(DEST/'all_seed_comparisons.csv')==read(DEST/'input_audit.json')['csv_hash']
    rows=list(csv.DictReader((DEST/'all_seed_comparisons.csv').open()))
    assert len(rows)==12
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(7.2,6.5))
    colors=['#0072B2','#D55E00'];markers=['o','s']
    for ti,task in enumerate(TASKS):
        subset=[r for r in rows if r['task']==task];assert len(subset)==6
        for metric,col in [('cost',0),('time',1)]:
            ax=axes[ti,col]
            for i,row in enumerate(subset):
                r=int(row['round']);mean=float(row[metric+'_change_percent']);lo=float(row[metric+'_lower_percent']);hi=float(row[metric+'_upper_percent'])
                assert np.isfinite([mean,lo,hi]).all()
                ax.hlines(i,lo,hi,color=colors[r],linewidth=1.4)
                ax.scatter(mean,i,color=colors[r],marker=markers[r],s=32,zorder=4)
                if metric=='time':
                    for repeat,marker,offset in [(0,'<',-.13),(1,'>',.13)]:
                        ax.scatter(float(row['time_repeat%d_percent'%repeat]),i+offset,facecolors='none',edgecolors=colors[r],marker=marker,s=30,zorder=3)
            ax.axvline(0,color='black',linewidth=.8,linestyle='--')
            ax.set_yticks(range(6));ax.set_yticklabels(['R%s / seed %s'%(r['round'],r['seed']) for r in subset])
            ax.set_ylim(5.6,-.6);ax.grid(axis='x',alpha=.2)
            if metric=='cost':
                ax.set_xscale('symlog',linthresh=2)
                ticks=[-100,0,10,1000] if task=='vehicle' else [-1,0,1,10]
                ax.set_xticks(ticks);ax.set_xticklabels([str(v) for v in ticks])
                ax.set_xlabel('Total cost change (%)\nSymmetric-log axis; linear within ±2%')
            else:
                ax.set_xlim(-30,30);ax.set_xticks([-20,-10,0,10,20]);ax.set_xlabel('Mean decision-time change (%)')
            ax.set_title('%s: %s'%(task.capitalize(),'total cost' if metric=='cost' else 'measured latency'),fontsize=10)
    fig.suptitle('Validation: both rounds fail the all-seed entry gates',fontsize=11,y=.985)
    handles=[Line2D([],[],color=colors[r],marker=markers[r],label='Round %d'%r) for r in range(2)]
    handles += [Line2D([],[],color='black',marker=m,markerfacecolor='none',linestyle='None',label='Timing repeat %d'%(j+1)) for j,m in enumerate(['<','>'])]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.10),ncol=4,frameon=False,fontsize=8)
    fig.text(.5,.02,'Negative change favors adaptation. Every seed is shown; 24 paired scenes per task.\nBars: conditional 95% scene-bootstrap intervals, not variability across timing sessions.\nSame fixed comparator for both labels (vehicle H25; pendulum H30). No test evidence.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.18,1,.96))
    preview=DEST/'cost_latency_preview.png';fig.savefig(preview,dpi=180)
    Image.open(preview).convert('L').save(DEST/'cost_latency_grayscale.png')
    issues=audit_layout(fig);write(DEST/'layout_audit.json',dict(issues=issues,rows_plotted=len(rows),source_hash=digest(Path(__file__))))
    if export:
        assert not issues,issues
        for suffix in ('pdf','svg'):fig.savefig(DEST/('cost_latency.'+suffix))
        fig.savefig(DEST/'cost_latency.png',dpi=300)
    plt.close(fig);print(json.dumps(dict(issues=issues,rows=len(rows))))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['prepare','render'],required=True);ap.add_argument('--skill-scripts',type=Path);ap.add_argument('--export',action='store_true');a=ap.parse_args()
    if a.mode=='prepare':prepare()
    else:render(a.skill_scripts,a.export)
