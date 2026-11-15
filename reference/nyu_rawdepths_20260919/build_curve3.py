"""Final: synthetic availability sweep vs two independent sources of REAL sensor failure.

lines      synthetic rect/multi/stripe block masks (band = spread over shapes)
stars      NYU test654 under its OWN rawDepths invalidity masks (scene-aligned)
circles    NYU test654 under Kinect-v1 masks transplanted from TUM fr1_desk
"""
import os; os.environ['MPLCONFIGDIR']='/tmp/claude-1000/mpl'
from pathlib import Path
import json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).parent
P='/home/lingxufeng/paper/Manuscript-polish-20260917/figures/fig5_test654_20260919'
SW=json.load(open(f'{P}/SWEEP_SUMMARY_test654.json'))
TU=json.load(open(f'{P}/tum_real_masks/TUM_MASK_RESULTS.json')); TU.pop('_pools',None)
NR=json.load(open(HERE/'NYU_RAW_RESULTS.json')); NR.pop('_strata',None); NR.pop('_hole',None)

SHAPES=['rect','multi','stripe']; AVAIL=[0,25,50,75,90,100]
def conds(a):
    if a==100: return ['clean']
    if a==0:   return ['missing']
    return [f'{s}{100-a}' for s in SHAPES]
ARMS=[('L0','RGB-only','#8f8f8f','--'),('C','C (conventional)','#c2703f','-'),('R','$R$ (ours)','#126879','-')]

plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral','Liberation Serif','DejaVu Serif'],
                     'mathtext.fontset':'stix','pdf.fonttype':42,'svg.fonttype':'none',
                     'axes.linewidth':0.6,'xtick.major.width':0.6,'ytick.major.width':0.6})
fig,ax=plt.subplots(figsize=(3.39,2.62))
ax.set_axisbelow(True); ax.grid(True,color='#e8e8e8',lw=0.6)
for sp in ('top','right'): ax.spines[sp].set_visible(False)
for sp in ('left','bottom'): ax.spines[sp].set_color('#9a9a9a')
ax.axvspan(60,80,color='#6a6a6a',alpha=.055,lw=0,zorder=0)

out={'availability':AVAIL,'synthetic':{},'real_nyu_rawdepths':{},'real_tum_transplant':{}}
for tag,lab,col,ls in ARMS:
    m=SW[tag]['miou']
    mean=[float(np.mean([m[c] for c in conds(a)])) for a in AVAIL]
    lo=[float(np.min([m[c] for c in conds(a)])) for a in AVAIL]
    hi=[float(np.max([m[c] for c in conds(a)])) for a in AVAIL]
    out['synthetic'][tag]={'mean':mean,'lo':lo,'hi':hi}
    ax.fill_between(AVAIL,lo,hi,color=col,alpha=.13,lw=0)
    ax.plot(AVAIL,mean,ls,color=col,lw=1.2,label=lab,zorder=3)
    xr=100*(1-NR[tag]['missing_fraction']['raw_aligned']); yr=NR[tag]['miou']['raw_aligned']
    out['real_nyu_rawdepths'][tag]={'availability':xr,'miou':yr}
    ax.plot([xr],[yr],'*',color=col,ms=9,mew=.6,mec='white',zorder=6)
    xt=[100*(1-TU[tag]['missing_fraction'][c]) for c in ('tum_p10','tum_p50','tum_p90')]
    yt=[TU[tag]['miou'][c] for c in ('tum_p10','tum_p50','tum_p90')]
    out['real_tum_transplant'][tag]={'availability':xt,'miou':yt}
    ax.plot(xt,yt,'o',color=col,ms=3.2,mew=.5,mec='white',alpha=.9,zorder=5)

ax.plot([],[],'*',color='#4a4a4a',ms=8,mew=.6,mec='white',label='real: NYU rawDepths')
ax.plot([],[],'o',color='#4a4a4a',ms=3.4,mew=.5,mec='white',label='real: TUM masks')
ax.set_xlabel('depth available at test time (%)',fontsize=7.5)
ax.set_ylabel('mIoU',fontsize=7.5)
ax.set_xticks(AVAIL); ax.tick_params(labelsize=7,length=2.5,color='#9a9a9a')
ax.set_xlim(-4,104); ax.set_ylim(47.4,54.8)
ax.legend(fontsize=6.3,frameon=False,loc='lower left',bbox_to_anchor=(0.005,-0.015),
          handlelength=1.7,labelspacing=.24)
ax.text(70,54.5,'real sensor\noperating band',fontsize=5.8,color='#6a6a6a',ha='center',va='top',linespacing=1.15)
ax.text(97,49.0,'lines: synthetic\nblock masks',fontsize=5.9,color='#8a8a8a',ha='right',linespacing=1.15)
fig.tight_layout(pad=0.25)
for e in ('png','pdf','svg'):
    fig.savefig(HERE/f'Fig_availability_synthetic_vs_real.{e}',dpi=400,bbox_inches='tight',pad_inches=.012)
(HERE/'Fig_availability_synthetic_vs_real_values.json').write_text(json.dumps(out,indent=2))
print('wrote', HERE/'Fig_availability_synthetic_vs_real.png')
