"""Render official T4 tables and plots from saved CSV only; never tune parameters."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import t4_official as t


def render(output):
    output=Path(output)
    if (output/'report_manifest.json').exists():raise FileExistsError('Report exists')
    config=json.loads((output/'config.json').read_text(encoding='utf-8'))
    key=str(Path(__file__).resolve().relative_to(t.a.REPO_DIR))
    if config['code_sha256'].get(key)!=t.a.sha(__file__):raise ValueError('Report renderer not frozen')
    detection=t.load_csv(output/'detection_summary.csv')
    summary=t.load_csv(output/'summary.csv')
    negatives=t.load_csv(output/'negative_summary.csv')
    all_det=[x for x in detection if x['level']=='all']
    all_restore=[x for x in summary if x['level']=='all']
    fig,axes=plt.subplots(1,3,figsize=(17,5.8))
    labels=[];values=[]
    for x in detection:
        if x['level']=='all':continue
        labels.append(f"{x['detector']}\nL{x['level']} (n={x['n_samples']})")
        values.append(float(x['mean_iou']))
    axes[0].bar(np.arange(len(values)),values,color=['#517fa4' if 'stamp_candidate' not in k else '#ce7536' for k in labels])
    axes[0].set_xticks(np.arange(len(values)),labels,rotation=65,ha='right',fontsize=7)
    axes[0].set_ylim(0,1);axes[0].set_ylabel('Mean per-image IoU');axes[0].set_title('Detection by level (different sources)')
    plotdata={'detection_labels':labels,'detection_iou':values,'restoration':[]}
    groups=sorted({(x['kind'],x['pen_color'],x['mask_source']) for x in all_restore})
    width=.18
    for panel,metric in zip(axes[1:],('mean_roi_mae','mean_text_proxy_mae')):
        for index,method in enumerate(t.r.METHODS):
            selected=[next(x for x in all_restore if (x['kind'],x['pen_color'],x['mask_source'])==g and x['method']==method) for g in groups]
            vals=[float(x[metric]) if x[metric] else np.nan for x in selected]
            panel.bar(np.arange(len(groups))+(index-1.5)*width,vals,width,label=method)
            plotdata['restoration'].append(dict(metric=metric,method=method,groups=[list(g) for g in groups],values=vals))
        names=[]
        for g in groups:
            row=next(x for x in all_restore if (x['kind'],x['pen_color'],x['mask_source'])==g)
            n=row['n_samples'] if metric=='mean_roi_mae' else row['n_text_proxy_samples']
            names.append('/'.join(g)+f'\nn={n}')
        panel.set_xticks(np.arange(len(groups)),names,rotation=55,ha='right',fontsize=7)
        panel.set_ylabel('Macro MAE (lower = closer to clean)');panel.legend(fontsize=7)
        panel.set_title('Fixed GT region' if metric=='mean_roi_mae' else 'Fixed GT & auto-Otsu text proxy')
    fig.suptitle('T4 official TEST image measurements | radius 3 | no CER | GT diagnostic is not an upper bound')
    fig.tight_layout();fig.savefig(output/'figure.png',dpi=170);fig.savefig(output/'figure.pdf');plt.close(fig)
    t.d.json_new(output/'plot_data.json',plotdata)
    lines=['# T4 official TEST: detection and restoration (no CER)','',
        'Frozen defaults, radius 3; 30 stamp +16 blue pen +14 black pen. Black prediction unsupported.',
        'All summaries are macro per image. GT-mask diagnostic is not a detector or a quality upper bound.',
        'Level strata use different sources. No tuning or OCR/CER; no method selected for deployment.','',
        '## Detection (all levels; per-level results in detection_summary.csv)','',
        '| Detector | n | IoU | Precision | Recall | Empty |','|---|---:|---:|---:|---:|---:|']
    for x in all_det:
        lines.append(f"|{x['detector']}|{x['n_samples']}|{float(x['mean_iou']):.6f}|{float(x['mean_precision']):.6f}|{float(x['mean_recall']):.6f}|{x['empty_predictions']}|")
    lines+=['','## Restoration (all levels; separate GT/predicted branches)','',
        '| Kind/color | Mask | Method | n | Text n | ROI MAE | Text-proxy MAE | Empty masks |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for x in all_restore:
        val=lambda k:f"{float(x[k]):.4f}" if x[k] else 'N/A'
        lines.append(f"|{x['kind']}/{x['pen_color']}|{x['mask_source']}|{x['method']}|{x['n_samples']}|{x['n_text_proxy_samples']}|{val('mean_roi_mae')}|{val('mean_text_proxy_mae')}|{x['empty_masks']}|")
    lines+=['','## Clean controls (injected-mark negatives, not certified free of original marks)','',
        '| Detector | n | Images with FP | FP pixels | Mean pixel FP ratio |','|---|---:|---:|---:|---:|']
    for x in negatives:lines.append(f"|{x['detector']}|{x['n_controls']}|{x['images_with_fp']}|{x['fp_pixels']}|{float(x['mean_fp_pixel_ratio']):.8f}|")
    lines+=['','## Interpretation limits','',
        '- GT support is injected alpha >0.15, not all visible color. Color recall can miss ink over black text.',
        '- Text proxy is automatic Otsu(clean), not human pixel labels. Empty text ROI stays N/A, not zero error.',
        '- ROI improvement can coincide with damage to text, especially GT-mask inpainting/white fill.',
        '- Same source split was used in T1-T3; dataset inventory/crops were already examined before this run.',
        '- Do not infer CER, OCR gains, general signature detection, real-image accuracy or T5 completion.','',
        '## Reproduce','',
        'From repo root (same frozen code, libraries, original data/5_synthetic and prior T4 evidence required):','',
        '```powershell',
        'code/.venv/Scripts/python.exe code/experiments/t4_official.py --output-dir results/t4/official_test_repeat',
        'code/.venv/Scripts/python.exe code/experiments/t4_official.py --verify-only --output-dir results/t4/official_test_repeat',
        'code/.venv/Scripts/python.exe code/experiments/t4_official_report.py --output-dir results/t4/official_test_repeat',
        '```','',
        'Existing output is never overwritten. Preserve history. On error the run aborts without completion marker;',
        'do not treat partial CSV/files as a completed run.','',
        'Data/images/results are outside Git: share the group dataset, step1_inventory.json, train_detection/config.json,',
        'train_val_restoration/config.json + completion.json and this result directory separately. A Git checkout alone',
        'does not include them. Frozen hashes in config.json verify code, previous configs and selected input files.',
        'Source is currently a working-tree snapshot: code hashes, not HEAD alone, identify the measured algorithms.','',
        'Artifacts: config.json (written before test decoding), detection.csv, detection_summary.csv,',
        'negative_controls.csv, negative_summary.csv, negative_restoration.csv (240 method/control rows),',
        'per_image.csv, summary.csv (overall and level/color strata), unsupported.csv, mask_cases.json, masks/,',
        'previews/ (native four-method comparisons and detector overlays), completion.json, verification.json.',
        'verification.json states exact replay scope; a subset replay is not a full rerun.','']
    (output/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    t.d.json_new(output/'report_manifest.json',dict(renderer_sha256=t.a.sha(__file__),
        source_csv_sha256={n:t.a.sha(output/n) for n in ('detection_summary.csv','summary.csv','negative_summary.csv')},
        outputs_sha256={n:t.a.sha(output/n) for n in ('figure.png','figure.pdf','plot_data.json','README.md')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=t.a.REPO_DIR/'results/t4/official_test_v1')
    render(p.parse_args().output_dir)
