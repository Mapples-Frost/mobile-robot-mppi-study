"""Audited training-only diagnostic figures; all seeds/scenarios, no efficacy claim."""
import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
from gated_horizon_search import OUT,TASKS
from paper_h_soft_probe import read,digest
from run import write

DEST=OUT/'training_figures'


def csvwrite(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def prepare(skill_scripts):
    source=OUT/'selected_training_diagnostics';report=read(source/'report.json')
    assert report['passed'] and report['completed_fits']==6 and not report['pending']
    for p,h in report['hashes'].items():
        assert digest(Path(p))==h
        for q,v in read(Path(p))['hashes'].items():assert digest(Path(q))==v
    full=read(OUT/'training_delivery/coverage_check.json');assert full['passed'] and full['conditions']==222
    for p,h in full['hashes'].items():assert digest(Path(p))==h
    pairs=list(csv.DictReader((source/'all_paired_training_episodes.csv').open()));assert len(pairs)==144
    rows=[];summary=[]
    for r in pairs:
        mode='failed' if r['fixed_success']=='False' or r['selected_success']=='False' else 'short' if int(r['short_steps']) else 'unchanged_h'
        rows.append(dict(task=r['task'],seed='S'+r['seed'],case='C%02d'%int(r['case']),raw_cost_delta=float(r['delta_total_cost']),
            physical_cost_delta=float(r['delta_physical_constraint_cost']),h_penalty_delta=float(r['delta_h_penalty']),
            short_steps=int(r['short_steps']),steps=int(r['steps']),short_step_percent=100*int(r['short_steps'])/int(r['steps']),mode=mode))
    for r in report['rows']:
        subset=[v for v in rows if v['task']==r['task'] and v['seed']=='S%d'%r['seed']]
        assert {v['case'] for v in subset}=={'C%02d'%i for i in range(24)} and len(subset)==24
        for column,key in [('raw_cost_delta','mean_delta_total_cost'),('physical_cost_delta','mean_delta_physical_constraint_cost'),('h_penalty_delta','mean_delta_h_penalty')]:
            np.testing.assert_allclose(np.mean([v[column] for v in subset]),r[key],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(r['mean_delta_total_cost'],r['mean_delta_physical_constraint_cost']+r['mean_delta_h_penalty'],rtol=1e-10,atol=1e-10)
        summary.append(dict(task=r['task'],seed='S%d'%r['seed'],policy=r['policy'],raw_cost_delta=r['mean_delta_total_cost'],
            physical_cost_delta=r['mean_delta_physical_constraint_cost'],h_penalty_delta=r['mean_delta_h_penalty'],
            cost_change_percent=100*(r['selected_raw_mean']-r['baseline_raw_mean'])/abs(r['baseline_raw_mean']),
            short_step_percent=100*r['short_step_fraction'],short_cases=r['short_cases'],cases=24,
            failed_cases=sum(v['mode']=='failed' for v in subset)))
    assert {(r['task'],r['seed']) for r in summary}=={(t,'S%d'%s) for t in TASKS for s in range(3)}
    DEST.mkdir(exist_ok=True);csvwrite(DEST/'all_cases.csv',rows);csvwrite(DEST/'all_models.csv',summary)
    for name in ('all_cases','all_models'):
        with (DEST/(name+'_profile.md')).open('w') as stream:
            subprocess.run([sys.executable,str(skill_scripts/'profile_data.py'),str(DEST/(name+'.csv')),'--group','task','--group','seed'],stdout=stream,stderr=subprocess.STDOUT,check=True)
    inputs=[source/'report.json',source/'all_paired_training_episodes.csv',OUT/'training_delivery/coverage_check.json',OUT/'audit_train.json']
    write(DEST/'input_audit.json',dict(passed=True,hashes={str(p):digest(p) for p in inputs},csv_hashes={name:digest(DEST/name) for name in ('all_cases.csv','all_models.csv')},
        rows=144,models=6,cases_per_model=24,question='How concentrated are selected training cost differences, and do they arise from physical costs or the horizon penalty?',
        design='Six-panel scenario scatter with all24 points per panel; companion per-model component dot plot and short-H fraction. Alternative: strip/box plot, but it loses scenario traceability. No connected scenario lines, uncertainty intervals or significance tests.',
        scope='Post-selection training description only. Scenarios differ between training seeds. Failure episodes remain. H penalty and short-H fraction are not measured latency. No validation/test outcomes accessed.',
        target='General internal research report;7.2-inch width,English labels plus Chinese caption,at least7.5pt text. No journal submission target.',source_hash=digest(Path(__file__))))
    print(json.dumps(dict(rows=144,models=6,range=[min(r['raw_cost_delta'] for r in rows),max(r['raw_cost_delta'] for r in rows)],
        ranges_by_task={t:[min(r['raw_cost_delta'] for r in rows if r['task']==t),max(r['raw_cost_delta'] for r in rows if r['task']==t)] for t in TASKS}),indent=2))


def render(skill_scripts,export=False):
    sys.path.insert(0,str(skill_scripts))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import ScalarFormatter
    from setup_style import setup_style
    from layout_tools import add_panel_labels
    from visual_qa import audit_layout
    from PIL import Image
    inputs=read(DEST/'input_audit.json');assert inputs['passed']
    for p,h in inputs['hashes'].items():assert digest(Path(p))==h
    for name,h in inputs['csv_hashes'].items():assert digest(DEST/name)==h
    assert all((DEST/(n+'_profile.md')).exists() for n in ('all_cases','all_models'))
    cases=list(csv.DictReader((DEST/'all_cases.csv').open()));models=list(csv.DictReader((DEST/'all_models.csv').open()))
    assert len(cases)==144 and len(models)==6
    setup_style(journal='general',lang='en',use_sciplots=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8.5,'axes.titlesize':9,
        'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.5,'figure.constrained_layout.use':False,
        'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    figures=[];fig,axes=plt.subplots(2,3,figsize=(7.2,6.5),sharex=True,sharey='row',constrained_layout=False)
    point_count=0
    styles={'short':dict(marker='o',color='#0072B2',label='Short H used'),
        'unchanged_h':dict(marker='o',facecolors='none',edgecolors='#888888',label='No short H; successful'),
        'failed':dict(marker='x',color='#222222',label='Failed episode (retained)')}
    for row,task in enumerate(TASKS):
        for seed in range(3):
            ax=axes[row,seed];group=[r for r in cases if r['task']==task and r['seed']=='S%d'%seed]
            model=next(r for r in models if r['task']==task and r['seed']=='S%d'%seed)
            for mode,style in styles.items():
                part=[r for r in group if r['mode']==mode];point_count+=len(part)
                ax.scatter([int(r['case'][1:]) for r in part],[float(r['raw_cost_delta']) for r in part],s=21,linewidths=.9,zorder=3,**{k:v for k,v in style.items() if k!='label'})
            ax.axhline(0,color='#777777',linewidth=.7,zorder=1);ax.grid(axis='y',alpha=.16)
            ax.set_yscale('symlog',linthresh=.05,linscale=1)
            ax.set_ylim((-50,2) if task=='vehicle' else (-3,.4))
            ax.set_yticks([-10,-1,-.1,0,.1,1] if task=='vehicle' else [-1,-.1,0,.1]);ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.set_xlim(-1,24);ax.set_xticks([0,6,12,18,23]);ax.set_title(task.capitalize()+' / S%d\n'%seed+model['policy'],pad=6)
            if seed==0:ax.set_ylabel('Raw total-cost difference\n(symmetric-log axis)')
            if row==1:ax.set_xlabel('Training scenario ID')
            if min(float(r['raw_cost_delta']) for r in group)<-5:
                extreme=sorted(group,key=lambda r:float(r['raw_cost_delta']))[:2]
                ax.text(.04,.95,'; '.join('%s: %.2f'%(r['case'],float(r['raw_cost_delta'])) for r in extreme),transform=ax.transAxes,fontsize=7.5,va='top')
    assert point_count==144
    fig.suptitle('Training only: all scenario-level cost differences',fontsize=11,y=.985)
    fig.text(.5,.934,'Selected gate minus paired fixed horizon; lower values favor the selected gate',ha='center',fontsize=8)
    handles=[Line2D([],[],linestyle='None',marker='o',color='#0072B2',label=styles['short']['label']),
        Line2D([],[],linestyle='None',marker='o',markerfacecolor='none',color='#888888',label=styles['unchanged_h']['label']),
        Line2D([],[],linestyle='None',marker='x',color='#222222',label=styles['failed']['label'])]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.125),ncol=3,frameon=False)
    fig.text(.5,.025,'Each panel has its own 24 training scenarios; all failures are included.\nAxes are linear within ±0.05; scales are shared within each task.\nPost-selection description only. Total cost includes an H penalty, not measured latency.',ha='center',fontsize=7.5)
    fig.tight_layout(rect=(.015,.19,.99,.91));add_panel_labels(fig,axes=list(axes.flat),fontsize=9,x_offset_pt=0,y_offset_pt=2,ha='left')
    figures.append(('case_differences',fig,dict(points=144,dimensions_inches=[7.2,6.5],axis='Symmetric-log with linear ±0.05; per-task limits; all points retained.')))
    fig,axes=plt.subplots(1,2,figsize=(7.2,4.5),gridspec_kw={'width_ratios':[1.35,1]},sharey=True,constrained_layout=False)
    component_styles=[('physical_cost_delta','#0072B2','^',-.16,'Physical + constraint'),('h_penalty_delta','#D55E00','s',0,'H penalty'),('raw_cost_delta','#222222','o',.16,'Total')]
    for i,r in enumerate(models):
        for column,color,marker,offset,label in component_styles:axes[0].scatter(float(r[column]),i+offset,color=color,marker=marker,s=25,zorder=3)
        fraction=float(r['short_step_percent']);axes[1].scatter(fraction,i,color='#0072B2',marker='D',s=25,zorder=3)
        axes[1].text(fraction+1,i,'%.2f%%'%fraction,va='center',fontsize=7.5)
    axes[0].axvline(0,color='#777777',linewidth=.8);axes[0].set_xlim(-1.95,.24);axes[0].set_xticks([-1.8,-1.2,-.6,0])
    axes[0].set_xlabel('Mean cost difference per training episode');axes[0].set_title('Cost components',pad=10)
    axes[1].set_xlim(0,31);axes[1].set_xticks([0,10,20,30]);axes[1].set_xlabel('Share of scored steps using short H (%)');axes[1].set_title('Short-H use',pad=10)
    axes[0].set_yticks(range(6));axes[0].set_yticklabels([r['task'].capitalize()+' / '+r['seed'] for r in models]);axes[0].set_ylim(5.5,-.5)
    for ax in axes:ax.grid(axis='x',alpha=.18);ax.axhline(2.5,color='#bbbbbb',linewidth=.7)
    fig.suptitle('Training only: components of the selected cost change',fontsize=11,y=.985)
    fig.legend(handles=[Line2D([],[],color=c,marker=m,linestyle='None',label=l) for _,c,m,_,l in component_styles],loc='lower center',bbox_to_anchor=(.5,.155),ncol=3,frameon=False)
    fig.text(.5,.025,'Each row is one fitted policy and its paired fixed baseline (24 training scenarios).\nMeans include failures; scenario-level differences are shown separately.\nNo confidence intervals: post-selection summaries. Neither H measure is actual latency.',ha='center',fontsize=7.5)
    fig.tight_layout(rect=(.005,.235,.995,.925));add_panel_labels(fig,axes=list(axes),fontsize=9,x_offset_pt=0,y_offset_pt=2,ha='left')
    figures.append(('cost_components',fig,dict(points=24,dimensions_inches=[7.2,4.5],axis='Linear; activation proportion starts at zero. Separate axes for cost and step share.')))
    outputs=[]
    for name,fig,meta in figures:
        preview=DEST/(name+'_preview.png');fig.savefig(preview,dpi=180)
        Image.open(preview).convert('L').save(DEST/(name+'_grayscale.png'))
        issues=audit_layout(fig);layout=dict(issues=issues,preview_hash=digest(preview),source_hash=digest(Path(__file__)),**meta);write(DEST/(name+'_layout.json'),layout)
        if export:
            assert not issues,issues
            review=read(DEST/'visual_review.json');assert review['passed'] and review['preview_hashes'][name]==digest(preview),'Read both actual color/grayscale previews before export'
            for suffix in ('pdf','svg'):fig.savefig(DEST/(name+'.'+suffix))
            fig.savefig(DEST/(name+'.png'),dpi=300)
        outputs.append(dict(name=name,issues=issues,exported=export));plt.close(fig)
    if export:
        shutil.copy2(Path(__file__),DEST/'plot_source.py')
        write(DEST/'exports.json',dict(training_only=True,hashes={str(p):digest(p) for p in DEST.glob('*') if p.suffix in ('.pdf','.svg','.png','.csv')},source_hash=digest(Path(__file__))))
    print(json.dumps(outputs,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['prepare','render'],required=True);ap.add_argument('--skill-scripts',type=Path,required=True);ap.add_argument('--export',action='store_true');a=ap.parse_args()
    if a.mode=='prepare':prepare(a.skill_scripts)
    else:render(a.skill_scripts,a.export)
