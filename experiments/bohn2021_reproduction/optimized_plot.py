"""Plot complete matched results and all fixed-H comparisons."""
import json
import numpy as np
from runtime import ART
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=ART/'results/optimized';DEST=ART/'report/optimized'


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for col,task in enumerate(['vehicle','pendulum']):
        rows=[r for r in summary['rows'] if r['task']==task and r['terminal']=='value']
        grid=sorted([r for r in rows if r['fixed_h'] is not None and r['seed']==0],key=lambda r:r['fixed_h'])
        ax=axs[0,col]
        ax.plot([r['fixed_h'] for r in grid],[r['mean_cost'] for r in grid],'o-',color='#4C72B0',label='Fixed H, seed 0')
        rl=[r for r in rows if r['fixed_h'] is None]
        for i,r in enumerate(rl):ax.axhline(r['mean_cost'],color='#C44E52',alpha=.45,lw=1,label='RL training seeds' if i==0 else None)
        ax.set(xlabel='Fixed horizon H',ylabel='Mean total cost (20 scenes)',title=task.capitalize()+'; learned terminal')
        ax.set_yscale('symlog',linthresh=10);ax.grid(alpha=.2);ax.legend(fontsize=8)
        c=summary['comparison'][task]['value'];h=c['selected_fixed_h']
        fixed=[r for r in rows if r['fixed_h']==h]
        x=np.mean([r['costs'] for r in fixed],axis=0);y=np.mean([r['costs'] for r in rl],axis=0)
        ax=axs[1,col];ax.scatter(x,y,color='#228833',s=25)
        lo=min(x.min(),y.min());hi=max(x.max(),y.max());ax.plot([lo,hi],[lo,hi],'--',color='#777777',lw=1)
        ax.set(xlabel='Fixed H=%d; mean over 3 seeds'%h,ylabel='RL; mean over 3 seeds',title='Paired holdout scenes; below line favors RL')
        ax.set_xscale('symlog',linthresh=10);ax.set_yscale('symlog',linthresh=10);ax.grid(alpha=.2)
    fig.suptitle('Improved reconstruction: unchanged paper reward, matched controller extensions',fontsize=13)
    for ext in ['png','pdf']:fig.savefig(DEST/('comparison.'+ext),dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
