"""Presentation-only T3 figure repair, preserving the original frozen run files."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,required=True)
    args=parser.parse_args()
    root=args.results_dir
    completion=json.loads((root/'completion.json').read_text(encoding='utf-8'))
    if (root/'figure_render_update.json').exists():
        raise FileExistsError('Figure already repaired; refusing to overwrite provenance')
    summary=root/'summary.csv'
    if sha(summary)!=completion['artifact_sha256']['summary.csv']:
        raise ValueError('Summary changed')
    for ext in ('png','pdf'):
        if sha(root/f'figure.{ext}')!=completion['artifact_sha256'][f'figure.{ext}']:
            raise ValueError('Initial figure changed')
        if (root/f'figure_initial.{ext}').exists():
            raise FileExistsError('Initial figure archive already exists')
    rows=list(csv.DictReader(summary.open(encoding='utf-8')))
    groups=['shadow_planar','shadow_silhouette','blur_gaussian','blur_motion','stamp']
    methods=['otsu','niblack','sauvola','wolf','gaussian']
    colors=dict(zip(methods,plt.get_cmap('tab10').colors[:5]))
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharey=True)
    for ax,group in zip(axes.flat,groups):
        for method in methods:
            for branch in ('none','t1'):
                values=sorted((r for r in rows if r['group']==group and r['method']==method
                               and r['branch']==branch),key=lambda r:int(r['level']))
                ax.plot([int(r['level']) for r in values],[float(r['mean_f1']) for r in values],
                        color=colors[method],linestyle='-' if branch=='none' else '--',
                        marker='o',markersize=3,label=f'{method} / {branch}')
        ax.set_title(group+(' (radius + opacity)' if group=='stamp' else ''))
        ax.set_xticks(sorted({int(r['level']) for r in rows if r['group']==group}))
        ax.set_ylim(0,1.02); ax.set_ylabel('Macro F1')
        ax.set_xlabel('Degradation level (0 = shared clean)'); ax.grid(alpha=.2)
    key=axes.flat[-1]; key.axis('off')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    key.legend(handles,labels,loc='upper left',bbox_to_anchor=(0,1),ncol=2,
               fontsize=9,frameon=False,columnspacing=1.2)
    key.text(.02,.12,'none: no shadow correction\nt1: Closing81 (frozen)\n30 paired sources per point\nAuto-Otsu reference; not OCR\nSample SD in summary.csv; no confidence bands',
             fontsize=9,transform=key.transAxes,va='bottom')
    fig.suptitle('T3: frozen T2 recipes on controlled variants of dataset 5 test sources')
    fig.tight_layout(rect=(0,0,1,.96))
    for ext in ('png','pdf'):
        shutil.copy2(root/f'figure.{ext}',root/f'figure_initial.{ext}')
        fig.savefig(root/f'figure.{ext}',dpi=180)
    plt.close(fig)
    report=dict(scope='Layout only: legend moved; all curve values unchanged',
                summary_sha256=sha(summary),renderer_sha256=sha(Path(__file__)),
                initial_figures={f'figure_initial.{e}':sha(root/f'figure_initial.{e}') for e in ('png','pdf')},
                current_figures={f'figure.{e}':sha(root/f'figure.{e}') for e in ('png','pdf')},
                frozen_config_sha256=sha(root/'config.json'))
    with (root/'figure_render_update.json').open('x',encoding='utf-8') as f:
        json.dump(report,f,indent=2)


if __name__=='__main__':
    main()
