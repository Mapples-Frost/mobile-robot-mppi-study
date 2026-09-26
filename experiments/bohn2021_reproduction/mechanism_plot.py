"""Standalone figures from saved, physically checked diagnostic trajectories."""
import json
import numpy as np
from runtime import ART
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

OUT=ART/'results/mechanism_probe'
DEST=ART/'report/mechanism_probe'


def load(p):return json.loads(p.read_text())


def main():
    DEST.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'savefig.dpi':180})
    case=load(OUT/'vehicle_bank.json')['cases'][6]
    base=load(OUT/'vehicle_case06/h10.json')
    restart=load(OUT/'solver_case06/30_left.json')
    trajectories={
        'Fixed H=10':base['trace'],
        'Fixed H=5':load(OUT/'vehicle_case06/h5.json')['trace'],
        'H=10, initial guess changed once':base['trace'][:30]+restart['trace']}
    colors=['#C44E52','#4C72B0','#228833']
    fig,axs=plt.subplots(2,2,figsize=(11.5,7.5),layout='constrained')
    for label,trace in trajectories.items():
        i=list(trajectories).index(label)
        xy=np.array([[base['initial_state']['x'],base['initial_state']['y']]]+[[r['state']['x'],r['state']['y']] for r in trace])
        axs[0,0].plot(xy[:,0],xy[:,1],color=colors[i],label=label,lw=1.7)
        axs[0,1].plot(np.arange(1,len(trace)+1)*.1,np.cumsum([r['cost'] for r in trace]),color=colors[i],lw=1.7)
        axs[1,0].plot(np.arange(1,len(trace)+1)*.1,[r['input']['u_s'] for r in trace],color=colors[i],lw=1.5)
    for j in range(3):
        def v(c):return case['tvp']['obj_%d_%s'%(j,c)][0]['true'][0]
        axs[0,0].add_patch(Circle((v('x'),v('y')),v('r'),facecolor='#999999',edgecolor='#444444',alpha=.6))
        axs[0,0].add_patch(Circle((v('x'),v('y')),1.5*v('r'),fill=False,edgecolor='#999999',ls=':',lw=.8))
    refs=np.array([[x['true'][0],y['true'][0]] for x,y in zip(case['tvp']['trajectory_x'],case['tvp']['trajectory_y'])])
    axs[0,0].plot(refs[:,0],refs[:,1],color='#666666',ls='--',lw=.9,label='Reference')
    axs[0,0].set(xlabel='x (m)',ylabel='y (m)',title='A  Same scene, different closed-loop routes')
    axs[0,0].set_aspect('equal',adjustable='datalim')
    axs[0,0].legend(fontsize=8,loc='lower left')
    axs[0,1].set(xlabel='Time after reset (s)',ylabel='Cumulative total cost',title='B  One initial-guess intervention at step 30')
    axs[1,0].set(xlabel='Time after reset (s)',ylabel='Forward speed (m/s)',title='C  Warm-start solution stops before obstacle')
    axs[1,0].axvline(3.,color='#444444',ls=':',lw=1)
    rows=[r for r in load(OUT/'solver_case06/completed.json')['rows'] if r['anchor']==30]
    for r in rows:
        axs[1,1].scatter(r['local']['objective'],r['suffix_cost'],s=55,color='#228833' if r['mode'] in ['left','right'] else '#C44E52')
    axs[1,1].annotate('warm / cold',(46.69,4552),xytext=(-85,-20),textcoords='offset points')
    axs[1,1].annotate('left / right',(32.61,27.69),xytext=(15,15),textcoords='offset points')
    axs[1,1].set(xlabel='NLP objective at the same state',ylabel='Remaining closed-loop cost',title='D  All four solves report success')
    axs[1,1].margins(.2)
    for ax in axs.flat:ax.grid(alpha=.18)
    fig.suptitle('Vehicle scene 6: horizon effects include local-solver history',fontsize=14)
    for ext in ['png','pdf']:fig.savefig(DEST/('solver_mechanism.'+ext))
    plt.close(fig)
    if not (OUT/'summary.json').exists():return
    summary=load(OUT/'summary.json')
    fig,axs=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for ax,task,title in zip(axs,['vehicle','pendulum'],['Vehicle','Inverted pendulum']):
        hs=list(range(5,51,5))
        rows={r['policy']:r for r in summary[task]['rows']}
        vals=np.array([rows['h%d'%h]['per_case'] for h in hs]).T
        for scene in vals:ax.plot(hs,scene,color='#AAAAAA',lw=.6,alpha=.5)
        ax.plot(hs,vals.mean(axis=0),color='#C44E52',marker='o',ms=3,label='Mean (10 scenes)')
        ax.plot(hs,np.median(vals,axis=0),color='#4C72B0',marker='s',ms=3,label='Median (10 scenes)')
        ax.set(xlabel='Fixed horizon H',ylabel='Total episode cost',title=title+'; zero terminal value')
        ax.set_yscale('symlog',linthresh=10)
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Scene sensitivity: individual scenes, mean and median (symlog cost axis)',fontsize=12)
    for ext in ['png','pdf']:fig.savefig(DEST/('fixed_h_sensitivity.'+ext))
    plt.close(fig)


if __name__=='__main__':main()
