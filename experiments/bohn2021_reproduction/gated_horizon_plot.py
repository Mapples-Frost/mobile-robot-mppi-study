"""Complete audited result preparation and all-seed cost/latency forest plots."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from gated_horizon_search import OUT,TASKS
from paper_h_soft_probe import read,digest
from run import write


def deduplicate(comparisons):
    assert len(comparisons)==12
    assert {(r['task'],r['seed'],r['comparator']) for r in comparisons}=={(t,s,c) for t in TASKS for s in range(3) for c in ('independent','matched')}
    groups={}
    for row in comparisons:
        key=(row['task'],row['seed'],row['adaptive']['path'],row['fixed']['path'])
        body={k:v for k,v in row.items() if k!='comparator'}
        if key in groups:
            assert body==groups[key]['body'],'Same recorded arms must give the same statistics'
            groups[key]['labels'].append(row['comparator'])
        else:groups[key]=dict(body=body,labels=[row['comparator']])
    return sorted(groups.values(),key=lambda x:(x['body']['task'],x['body']['seed'],x['labels']))


def prepare(split,skill_scripts):
    output=OUT/(split+'_figures');report=OUT/(split+'_delivery/effect_gate.json');data=read(report)
    for p,h in data['hashes'].items():assert digest(Path(p))==h
    timing_audit=read(OUT/('audit_timing_'+split+'.json'));assert timing_audit['passed']
    for p,h in timing_audit['hashes'].items():assert digest(Path(p))==h
    rows=[]
    for group in deduplicate(data['comparisons']):
        a=group['body'];assert a['timing'] is not None
        scale=max(abs(a['fixed']['total_cost']),1e-12);tm=a['timing'];assert len(tm['repeat_ratios'])==2
        rows.append(dict(task=a['task'],seed='S%d'%a['seed'],comparator='+'.join(sorted(group['labels'])),fixed_h=a['fixed_h'],
            cost_change_percent=100*a['relative_cost_difference'],cost_lower_percent=100*a['cost_difference']['lower']/scale,cost_upper_percent=100*a['cost_difference']['upper']/scale,
            time_change_percent=100*(tm['mean_ratio']-1),time_lower_percent=100*(tm['lower']-1),time_upper_percent=100*(tm['upper']-1),
            time_repeat0_percent=100*(tm['repeat_ratios'][0]-1),time_repeat1_percent=100*(tm['repeat_ratios'][1]-1),
            safety_control_adaptation=bool(a['safe'] and a['control_noninferior'] and a['adapted']),
            adaptive_success=a['adaptive']['success'],fixed_success=a['fixed']['success'],adaptive_constraints=a['adaptive']['constraints'],fixed_constraints=a['fixed']['constraints'],
            adaptive_initial_failures=a['adaptive']['initial_failures'],fixed_initial_failures=a['fixed']['initial_failures'],adaptive_final_failures=a['adaptive']['final_failures'],fixed_final_failures=a['fixed']['final_failures']))
    output.mkdir(exist_ok=True);csvpath=output/'all_seed_comparisons.csv'
    with csvpath.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    profile=output/'data_profile.md'
    with profile.open('w') as stream:
        subprocess.run([sys.executable,str(skill_scripts/'profile_data.py'),str(csvpath),'--group','task','--group','comparator'],stdout=stream,stderr=subprocess.STDOUT,check=True)
    effect=data['validation_effect_passed'] if split=='validation' else data['independent_effect_passed']
    write(output/'input_audit.json',dict(source=str(report),source_hash=digest(report),csv_hash=digest(csvpath),rows=len(rows),original_comparisons=12,split=split,cases=32 if split=='validation' else 64,registered_effect_passed=effect,
        all_seeds_retained=True,duplicate_rule='Merge comparator labels only when exact same adaptive/fixed paths and every statistic agree. Labels are not independent samples.',
        question='Do all trained seeds improve raw control-cost/actual decision-latency tradeoffs against both selected fixed comparators?',
        plot='Forest point+interval per seed/comparator, tasks separated, both measured timing-repeat points shown. Alternative: paired-scene scatter if heterogeneity warrants an additional panel. General internal-report dimensions; no specified journal.',
        profile_notes='Seed is categorical. Means, endpoints and repeats are estimates of the same outcomes, not independent observations; their correlation is not a scientific result. Inspect range before choosing linear or symmetric-log cost axis.',
        interval='Conditional95% paired-scene percentile bootstrap,10000 draws. Two timing repeats are not independent scenes or a confidence interval across host sessions. No multiplicity-adjusted discovery claim.'))
    print(str(csvpath))


def render(split,skill_scripts,cost_scale,export):
    sys.path.insert(0,str(skill_scripts));from visual_qa import audit_layout
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator,ScalarFormatter
    from PIL import Image
    dest=OUT/(split+'_figures');audit=read(dest/'input_audit.json')
    assert digest(Path(audit['source']))==audit['source_hash'];assert digest(dest/'all_seed_comparisons.csv')==audit['csv_hash']
    assert (dest/'data_profile.md').exists()
    rows=list(csv.DictReader((dest/'all_seed_comparisons.csv').open()));assert len(rows)==audit['rows']
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(7.2,6.5));styles={'independent':('#0072B2','o','I'),'matched':('#D55E00','s','M'),'independent+matched':('#333333','D','I=M')}
    for ti,task in enumerate(TASKS):
        subset=[r for r in rows if r['task']==task];assert {r['seed'] for r in subset}=={'S0','S1','S2'}
        for metric,col in [('cost',0),('time',1)]:
            ax=axes[ti,col];extent=[0.]
            for i,row in enumerate(subset):
                color,marker,_=styles[row['comparator']];mean,lo,hi=[float(row[metric+'_'+k+'_percent']) for k in ('change','lower','upper')]
                assert np.isfinite([mean,lo,hi]).all() and lo<=hi
                ax.hlines(i,lo,hi,color=color,linewidth=1.4);ax.scatter(mean,i,color=color,marker=marker,s=30,zorder=4);extent.extend([mean,lo,hi])
                if metric=='time':
                    for repeat,m,offset in [(0,'<',-.13),(1,'>',.13)]:
                        value=float(row['time_repeat%d_percent'%repeat]);extent.append(value)
                        ax.scatter(value,i+offset,facecolors='none',edgecolors=color,marker=m,s=26,zorder=3)
            ax.axvline(0,color='black',linestyle='--',linewidth=.8);ax.set_yticks(range(len(subset)))
            ax.set_yticklabels([r['seed']+' / '+styles[r['comparator']][2]+' H'+r['fixed_h']+(' !' if r['safety_control_adaptation']=='False' else '') for r in subset])
            ax.set_ylim(len(subset)-.4,-.6);ax.grid(axis='x',alpha=.2)
            if metric=='cost' and cost_scale=='symlog':
                ax.set_xscale('symlog',linthresh=2);ax.set_xlabel('Raw total-cost change (%)\nSymmetric-log; linear within ±2%')
                if max(abs(v) for v in extent)<=2:ax.xaxis.set_major_locator(MaxNLocator(5))
                ax.xaxis.set_major_formatter(ScalarFormatter())
                # Autoscale retains all means/endpoints; tick spacing is checked before export.
            else:
                span=max(max(extent)-min(extent),2.);ax.set_xlim(min(extent)-.08*span,max(extent)+.08*span);ax.xaxis.set_major_locator(MaxNLocator(5))
                ax.set_xlabel('Raw total-cost change (%)' if metric=='cost' else 'Measured decision-time change (%)')
            ax.set_title(task.capitalize()+': '+('cost' if metric=='cost' else 'latency'),fontsize=10)
    title=split.capitalize()+': all seeds and selected fixed baselines'
    fig.suptitle(title,fontsize=11,y=.985)
    present=sorted({r['comparator'] for r in rows});handles=[Line2D([],[],color=styles[k][0],marker=styles[k][1],label=styles[k][2]) for k in present]
    handles += [Line2D([],[],color='black',marker=m,markerfacecolor='none',linestyle='None',label='Timing repeat %d'%(i+1)) for i,m in enumerate(['<','>'])]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.115),ncol=len(handles),frameon=False,fontsize=8)
    fig.text(.5,.02,'Negative change favors adaptation; %d paired scenes per task; conditional 95%% intervals.\nI: independent terminal; M: matched terminal. !: safety/control/adaptation prerequisite fails.\nRegistered effect gate: %s. H is not a speed measure; two repeats are not extra scenes.'%(audit['cases'],'passed' if audit['registered_effect_passed'] else 'failed'),ha='center',fontsize=7.5)
    fig.tight_layout(rect=(0,.19,1,.96));preview=dest/'cost_latency_preview.png';fig.savefig(preview,dpi=180)
    Image.open(preview).convert('L').save(dest/'cost_latency_grayscale.png')
    issues=audit_layout(fig);write(dest/'layout_audit.json',dict(issues=issues,rows=len(rows),cost_scale=cost_scale,source_hash=digest(Path(__file__)),preview_hash=digest(preview)))
    if export:
        assert not issues,issues
        review=read(dest/'visual_review.json');assert review['passed'] and review['preview_hash']==digest(preview),'Inspect actual color/grayscale preview before vector export'
        for suffix in ('pdf','svg'):fig.savefig(dest/('cost_latency.'+suffix))
        fig.savefig(dest/'cost_latency.png',dpi=300)
    plt.close(fig);print(json.dumps(dict(rows=len(rows),issues=issues,exported=export)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['prepare','render'],required=True);ap.add_argument('--split',choices=['validation','test'],default='validation')
    ap.add_argument('--skill-scripts',type=Path,required=True);ap.add_argument('--cost-scale',choices=['linear','symlog'],default='linear');ap.add_argument('--export',action='store_true');a=ap.parse_args()
    if a.mode=='prepare':prepare(a.split,a.skill_scripts)
    else:render(a.split,a.skill_scripts,a.cost_scale,a.export)
