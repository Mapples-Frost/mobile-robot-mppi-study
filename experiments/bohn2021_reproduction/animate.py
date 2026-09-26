"""Render recorded physical states, never interpolate a fictitious controller run."""
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from matplotlib.patches import Circle,Rectangle
ROOT=Path(__file__).resolve().parents[2]
ART=ROOT/'research_artifacts/bohn2021_reproduction_2026-09-17'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--group',default='full',choices=['full','refined','paper_defaults']);args=ap.parse_args()
    mode='eval_value' if args.group=='full' else 'holdout_value'
    bank_name='test' if args.group=='full' else 'holdout'
    offset=1 if args.group=='full' else 2
    report=ART/'report' if args.group=='full' else ART/'report'/('diagnosis' if args.group=='refined' else 'paper_defaults')
    report.mkdir(parents=True,exist_ok=True)
    for task in ['pendulum','vehicle']:
        folder=ART/'results'/args.group/(task+'_rl_s0')
        if not (folder/'completed.json').exists():continue
        t=json.loads((folder/mode/'trace_00.json').read_text())
        case=json.loads((ART/'configs'/(task+'_'+bank_name+'_bank.json')).read_text())['cases'][0]
        fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)
        label=ax.text(.02,.96,'',transform=ax.transAxes,va='top')
        if task=='pendulum':
            ax.plot([-1.5,1.5],[0,0],color='#64748b',lw=3)
            cart=Rectangle((0,0),.2,.08,color='#2563eb');ax.add_patch(cart)
            rod,=ax.plot([],[],color='#d97706',lw=4,marker='o')
            ref=ax.axvline(0,color='#16a34a',ls='--')
            ax.set(xlim=(-1.6,1.6),ylim=(-.12,.5),xlabel='Cart position (m)',ylabel='Height (m)')
            def update(i):
                s=t[i]['state'];x=s['pos'];theta=s['theta'];cart.set_x(x-.1)
                rod.set_data([x,x+.25*np.sin(theta)],[.08,.08+.25*np.cos(theta)])
                r=case['tvp']['pos_r'][i+offset]['true'][0];ref.set_xdata([r,r])
                label.set_text('t=%.2f s | H=%d | angle=%.1f°'%((i+offset)*.04,t[i]['horizon'],np.degrees(theta)))
        else:
            tvp=case['tvp'];ns=case['reference']['traj_steps']
            rx=[v['true'][0] for v in tvp['trajectory_x'][:ns]];ry=[v['true'][0] for v in tvp['trajectory_y'][:ns]]
            ax.plot(rx,ry,'--',color='#94a3b8')
            for j in range(3):
                x,y,r=[tvp['obj_%d_%s'%(j,n)][0]['true'][0] for n in ['x','y','r']]
                ax.add_patch(Circle((x,y),r,color='#64748b',alpha=.5))
                ax.add_patch(Circle((x,y),1.5*r,fill=False,ls=':',edgecolor='#64748b'))
            path,=ax.plot([],[],color='#2563eb');point,=ax.plot([],[],'o',color='#2563eb')
            ax.set(xlim=(min(rx)-2,max(rx)+2),ylim=(min(ry)-3,max(ry)+3),xlabel='x (m)',ylabel='y (m)');ax.set_aspect('equal')
            def update(i):
                path.set_data([r['state']['x'] for r in t[:i+1]],[r['state']['y'] for r in t[:i+1]])
                point.set_data([t[i]['state']['x']],[t[i]['state']['y']])
                label.set_text('t=%.1f s | H=%d'%((i+offset)*.1,t[i]['horizon']))
        ax.set_title(task.capitalize()+' | RL seed 0, frozen case 0, learned terminal value')
        fps=25 if task=='pendulum' else 10
        anim=FuncAnimation(fig,update,frames=len(t),interval=1000/fps)
        dest=report/(task+'_replay.gif');anim.save(str(dest),writer=PillowWriter(fps=fps),dpi=90)
        plt.close(fig);print(dest)


if __name__=='__main__':main()
