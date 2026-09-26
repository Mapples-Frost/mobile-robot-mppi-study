"""Preselected holdout case 0: all RL seeds, fixed comparators, actual horizons."""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from runtime import ART

ap=argparse.ArgumentParser();ap.add_argument('--group',default='paper_defaults',choices=['paper_defaults','refined']);args=ap.parse_args()
src=ART/'results'/args.group
out=ART/'report'/('paper_defaults' if args.group=='paper_defaults' else 'diagnosis')
out.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
colors=['#2878B5','#D47932','#3C9A70']
for col,task in enumerate(['vehicle','pendulum']):
    case=json.loads((ART/'configs'/(task+'_holdout_bank.json')).read_text())['cases'][0]
    ax,hax=axes[0,col],axes[1,col]
    fixed=10 if task=='vehicle' else 30
    for i,name in enumerate([task+'_rl_s%d'%s for s in range(3)]+[task+'_fixed_h%d'%fixed]):
        data=json.loads((src/name/'holdout_value/trace_00.json').read_text())
        label='RL seed %d'%i if i<3 else 'Fixed H%d'%fixed
        color=colors[i] if i<3 else '#505050';ls='-' if i<3 else '--'
        t=(np.arange(len(data))+2)*(.1 if task=='vehicle' else .04)
        if task=='vehicle':ax.plot([r['state']['x'] for r in data],[r['state']['y'] for r in data],label=label,color=color,ls=ls,lw=1.6)
        else:ax.plot(t,[r['state']['pos'] for r in data],label=label,color=color,ls=ls,lw=1.5)
        hax.step(t,[r['horizon'] for r in data],where='post',label=label,color=color,ls=ls,lw=1.3)
    if task=='vehicle':
        tvp=case['tvp'];n=case['reference']['traj_steps']
        ax.plot([v['true'][0] for v in tvp['trajectory_x'][:n]],[v['true'][0] for v in tvp['trajectory_y'][:n]],':',color='#9D9D9D',label='Reference')
        for j in range(3):
            x,y,r=[tvp['obj_%d_%s'%(j,c)][0]['true'][0] for c in ['x','y','r']]
            ax.add_patch(Circle((x,y),r,color='#909090',alpha=.5))
            ax.add_patch(Circle((x,y),1.5*r,fill=False,ls=':',edgecolor='#909090'))
        ax.set(xlabel='x (m)',ylabel='y (m)',title='Vehicle: held-out case 0');ax.set_aspect('equal',adjustable='datalim')
    else:
        t=np.arange(102)*.04
        ax.step(t,[v['true'][0] for v in case['tvp']['pos_r'][:102]],where='post',color='#999999',ls=':',label='Reference')
        ax.axhline(1.5,color='#B25050',lw=.8,ls=':');ax.axhline(-1.5,color='#B25050',lw=.8,ls=':')
        ax.set(xlabel='Physical time (s)',ylabel='Cart position (m)',title='Pendulum: held-out case 0')
    ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.15)
    hax.set(xlabel='Physical time (s)',ylabel='Executed horizon H',ylim=(0,51))
    hax.grid(alpha=.15)
fig.savefig(out/'case0_trajectories.png');fig.savefig(out/'case0_trajectories.pdf');plt.close(fig)
print(out/'case0_trajectories.png')
