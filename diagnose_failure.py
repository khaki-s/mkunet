"""Diagnose WT failure mode for a single case: false-positive vs false-negative driven.

Usage:
    python diagnose_failure.py                      # default case 00493
    python diagnose_failure.py 00493 00642 00242    # several cases
    python diagnose_failure.py --all                # rank all test cases by WT dice

Reads the already-written predictions (no GPU / no inference needed).
"""
import sys
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

PRED_DIR = ('/root/autodl-tmp/data/nnUNet_results/Dataset137_BraTS2021/'
            'nnUNetTrainer_FADSR_100epochs__nnUNetPlans__2d/fold_0/predictions_2d_test')
GT_DIR = '/root/autodl-tmp/data/nnUNet_raw/Dataset137_BraTS2021/labelsTs'
IMG_DIR = '/root/autodl-tmp/data/nnUNet_raw/Dataset137_BraTS2021/imagesTs'
FLAIR_SUFFIX = '_0003.nii.gz'   # channel_names: 0=T1 1=T1ce 2=T2 3=Flair

# BraTS label ids -> evaluated regions (from dataset.json)
REGIONS = {'WT': (1, 2, 3), 'TC': (2, 3), 'ET': (3,)}


def dice(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    s = pred.sum() + gt.sum()
    return 1.0 if s == 0 else 2.0 * inter / s


def load(case):
    p = sitk.ReadImage(f'{PRED_DIR}/{case}.nii.gz')
    g = sitk.ReadImage(f'{GT_DIR}/{case}.nii.gz')
    spacing = p.GetSpacing()                      # (x, y, z); array axes are (z, y, x)
    vox_ml = float(np.prod(spacing)) / 1000.0     # mm^3 -> mL
    return (sitk.GetArrayFromImage(p).astype(np.uint8),
            sitk.GetArrayFromImage(g).astype(np.uint8), vox_ml)


def region_mask(lab, ids):
    return np.isin(lab, ids)


def confusion(pred_m, gt_m, vox_ml):
    tp = int(np.logical_and(pred_m, gt_m).sum())
    fp = int(np.logical_and(pred_m, ~gt_m).sum())
    fn = int(np.logical_and(~pred_m, gt_m).sum())
    return dict(tp=tp, fp=fp, fn=fn,
                dice=dice(pred_m, gt_m),
                # recall/precision make the FP-vs-FN story explicit
                recall=tp / max(tp + fn, 1),
                precision=tp / max(tp + fp, 1),
                pred_ml=pred_m.sum() * vox_ml,
                gt_ml=gt_m.sum() * vox_ml)


def components(mask, vox_ml):
    """Connected components of a binary mask, largest first."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return lab, []
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    order = np.argsort(-sizes)
    return lab, [(int(order[i]) + 1, float(sizes[order[i]] * vox_ml)) for i in range(n)]


def diagnose(case, verbose=True):
    pred, gt, vox_ml = load(case)
    out = {'case': case}

    if verbose:
        print(f'\n{"="*74}\n{case}   volume {pred.shape}   voxel {vox_ml*1000:.2f} mm^3\n{"="*74}')
        print('--- per-region confusion (WT is the one under investigation) ---')
        print(f'{"reg":4} {"dice":>7} {"recall":>7} {"prec":>7} {"TP":>8} {"FP":>8} {"FN":>8} '
              f'{"pred mL":>8} {"gt mL":>8}')

    for name, ids in REGIONS.items():
        c = confusion(region_mask(pred, ids), region_mask(gt, ids), vox_ml)
        out[name] = c
        if verbose:
            print(f'{name:4} {c["dice"]:7.3f} {c["recall"]:7.3f} {c["precision"]:7.3f} '
                  f'{c["tp"]:8,} {c["fp"]:8,} {c["fn"]:8,} {c["pred_ml"]:8.1f} {c["gt_ml"]:8.1f}')

    wt_p = region_mask(pred, REGIONS['WT'])
    wt_g = region_mask(gt, REGIONS['WT'])
    c = out['WT']

    # ---- the actual verdict -------------------------------------------------
    if verbose:
        print('\n--- WT failure mode ---')
        ratio = c['fp'] / max(c['fn'], 1)
        if c['fp'] + c['fn'] == 0:
            verdict = 'no error'
        elif ratio > 3:
            verdict = f'FALSE-POSITIVE driven (FP/FN = {ratio:.1f})'
        elif ratio < 1 / 3:
            verdict = f'FALSE-NEGATIVE driven (FN/FP = {1/max(ratio,1e-9):.1f})'
        else:
            verdict = f'MIXED (FP/FN = {ratio:.1f})'
        print(f'  FP={c["fp"]:,} voxels ({c["fp"]*vox_ml:.1f} mL)   '
              f'FN={c["fn"]:,} voxels ({c["fn"]*vox_ml:.1f} mL)')
        print(f'  => {verdict}')
        print(f'  predicted WT volume {c["pred_ml"]:.1f} mL vs ground truth {c["gt_ml"]:.1f} mL '
              f'({c["pred_ml"]/max(c["gt_ml"],1e-9):.2f}x)')

    # ---- connected components: is the error one blob far away, or diffuse? ---
    lab_p, comps_p = components(wt_p, vox_ml)
    lab_g, comps_g = components(wt_g, vox_ml)
    if verbose:
        print(f'\n--- connected components (WT) ---')
        print(f'  prediction: {len(comps_p)} component(s), largest {comps_p[0][1]:.1f} mL '
              f'= {100*comps_p[0][1]/max(c["pred_ml"],1e-9):.1f}% of predicted volume'
              if comps_p else '  prediction: empty')
        print(f'  ground truth: {len(comps_g)} component(s)'
              + (f', largest {comps_g[0][1]:.1f} mL' if comps_g else ''))
        for i, (cid, ml) in enumerate(comps_p[:6]):
            comp = lab_p == cid
            ov = np.logical_and(comp, wt_g).sum() / max(comp.sum(), 1)
            print(f'    pred comp {i}: {ml:7.1f} mL   overlap-with-GT {100*ov:5.1f}%'
                  f'   {"(mostly true)" if ov > 0.5 else "(mostly FALSE POSITIVE)"}')

    # ---- would largest-component postprocessing fix it? ---------------------
    if comps_p:
        biggest = lab_p == comps_p[0][0]
        out['dice_lcc'] = dice(biggest, wt_g)
        if verbose:
            print(f'\n--- largest-connected-component postprocessing ---')
            print(f'  WT dice {c["dice"]:.3f} -> {out["dice_lcc"]:.3f} '
                  f'({out["dice_lcc"]-c["dice"]:+.3f})')

    # ---- where do the errors sit, and does FLAIR support them? -------------
    try:
        fl = sitk.GetArrayFromImage(sitk.ReadImage(f'{IMG_DIR}/{case}{FLAIR_SUFFIX}')).astype(np.float32)
        brain = fl > np.percentile(fl[fl > 0], 1) if (fl > 0).any() else fl > 0
        # robust normalisation inside the brain so numbers are comparable across cases
        mu, sd = fl[brain].mean(), fl[brain].std() + 1e-6
        z = (fl - mu) / sd
        fp_m = np.logical_and(wt_p, ~wt_g)
        fn_m = np.logical_and(~wt_p, wt_g)
        tp_m = np.logical_and(wt_p, wt_g)
        if verbose:
            print(f'\n--- FLAIR intensity (z-score within brain) ---')
            for nm, m in (('true positive', tp_m), ('false positive', fp_m),
                          ('false negative', fn_m), ('true negative brain', np.logical_and(brain, ~wt_g) & ~wt_p)):
                if m.any():
                    print(f'  {nm:22s} n={int(m.sum()):8,}  FLAIR z mean={z[m].mean():+.2f} '
                          f'median={np.median(z[m]):+.2f}')
            if fn_m.any() and tp_m.any():
                print(f'  -> missed voxels are {z[tp_m].mean()-z[fn_m].mean():+.2f} z '
                      f'{"DIMMER" if z[fn_m].mean() < z[tp_m].mean() else "BRIGHTER"} than the '
                      f'ones it got right')
    except Exception as e:
        if verbose:
            print(f'\n  (FLAIR check skipped: {type(e).__name__}: {e})')

    # ---- slice profile: is the failure localised to part of the volume? ------
    if verbose:
        print(f'\n--- axial slice profile (2D model predicts slice by slice) ---')
        gt_per = wt_g.sum(axis=(1, 2))
        pr_per = wt_p.sum(axis=(1, 2))
        nz = np.where(gt_per > 0)[0]
        if len(nz):
            print(f'  GT spans slices {nz[0]}..{nz[-1]} ({len(nz)} slices)')
            miss = [i for i in nz if pr_per[i] == 0]
            print(f'  slices with GT but EMPTY prediction: {len(miss)}'
                  + (f'  e.g. {miss[:12]}' if miss else ''))
            extra = [i for i in range(len(gt_per)) if gt_per[i] == 0 and pr_per[i] > 0]
            print(f'  slices with prediction but NO GT:    {len(extra)}'
                  + (f'  e.g. {extra[:12]}' if extra else ''))
            worst = sorted(nz, key=lambda i: dice(wt_p[i], wt_g[i]))[:5]
            print('  worst slices (per-slice WT dice):')
            for i in worst:
                print(f'    slice {i:3d}: dice={dice(wt_p[i], wt_g[i]):.3f} '
                      f'gt={int(gt_per[i]):6,} pred={int(pr_per[i]):6,}')
    return out


def rank_all(region='WT', topn=12):
    """Rank every test case by Dice for one region and summarise the failure mode."""
    import glob, os
    ids = REGIONS[region]
    cases = sorted(os.path.basename(f).replace('.nii.gz', '')
                   for f in glob.glob(f'{PRED_DIR}/*.nii.gz'))
    rows = []
    for cs in cases:
        try:
            pred, gt, vox = load(cs)
            c = confusion(region_mask(pred, ids), region_mask(gt, ids), vox)
            lab_p, comps = components(region_mask(pred, ids), vox)
            lcc = dice(lab_p == comps[0][0], region_mask(gt, ids)) if comps else 0.0
            rows.append((cs, c['dice'], c['recall'], c['precision'], c['fp'], c['fn'],
                         len(comps), lcc, c['gt_ml'], c['pred_ml']))
        except Exception as e:
            print(f'  {cs}: {type(e).__name__}')
    rows.sort(key=lambda r: r[1])

    def mode_of(r):
        return 'FP' if r[4] > 3 * max(r[5], 1) else ('FN' if r[5] > 3 * max(r[4], 1) else 'mixed')

    print(f'\n########## region = {region}  (labels {ids}) ##########')
    print(f'{"case":18} {"dice":>7} {"recall":>7} {"prec":>7} {"FP":>8} {"FN":>8} '
          f'{"nCC":>4} {"gt mL":>8} {"pred mL":>8} {"mode":>6}')
    for r in rows[:topn]:
        print(f'{r[0]:18} {r[1]:7.3f} {r[2]:7.3f} {r[3]:7.3f} {r[4]:8,} {r[5]:8,} '
              f'{r[6]:4d} {r[8]:8.1f} {r[9]:8.1f} {mode_of(r):>6}')

    d = np.array([r[1] for r in rows])
    l = np.array([r[7] for r in rows])
    rec = np.array([r[2] for r in rows]); pre = np.array([r[3] for r in rows])
    print(f'\n  mean={d.mean():.4f}  median={np.median(d):.4f}  '
          f'-> largest-CC {l.mean():.4f} ({l.mean()-d.mean():+.4f})')
    k = 10
    deficit_all = (np.median(d) - d).clip(0).sum()
    deficit_k = (np.median(d) - np.sort(d)[:k]).clip(0).sum()
    print(f'  worst {k} cases hold {100*deficit_k/max(deficit_all,1e-9):.0f}% of the '
          f'below-median deficit; lifting them to median gives '
          f'{d.mean()+deficit_k/len(d):.4f} ({deficit_k/len(d):+.4f})')
    fp_d = sum(1 for r in rows if mode_of(r) == 'FP')
    fn_d = sum(1 for r in rows if mode_of(r) == 'FN')
    print(f'  failure mode over {len(rows)} cases: FP-driven {fp_d}, FN-driven {fn_d}, '
          f'mixed {len(rows)-fp_d-fn_d}')
    w = np.argsort(d)[:k]
    print(f'  worst {k}: recall={rec[w].mean():.3f} prec={pre[w].mean():.3f}   '
          f'rest: recall={np.delete(rec,w).mean():.3f} prec={np.delete(pre,w).mean():.3f}')
    print(f'  worst {k} by dice: ' + ', '.join(r[0].replace("BraTS2021_", "") for r in rows[:k]))
    return {r[0]: r[1] for r in rows}


def rank_all_regions():
    """Rank all three regions and report how much their worst cases overlap."""
    per = {reg: rank_all(reg) for reg in ('WT', 'TC', 'ET')}
    print('\n########## overlap of the worst 10 across regions ##########')
    worst = {reg: set(sorted(d, key=lambda c: d[c])[:10]) for reg, d in per.items()}
    for a, b in (('WT', 'TC'), ('WT', 'ET'), ('TC', 'ET')):
        inter = worst[a] & worst[b]
        print(f'  {a} n {b}: {len(inter)}/10  '
              + (', '.join(sorted(x.replace("BraTS2021_", "") for x in inter)) if inter else ''))
    allthree = worst['WT'] & worst['TC'] & worst['ET']
    print(f'  in all three: {len(allthree)}  '
          + ', '.join(sorted(x.replace("BraTS2021_", "") for x in allthree)))
    print('\n  cases that are bad at WT but FINE at TC/ET '
          '(WT-specific failure -> what we want to fix):')
    for c in sorted(worst['WT'], key=lambda x: per['WT'][x]):
        if c not in worst['TC'] and c not in worst['ET']:
            print(f'    {c.replace("BraTS2021_","")}: WT={per["WT"][c]:.3f} '
                  f'TC={per["TC"][c]:.3f} ET={per["ET"][c]:.3f}')


if __name__ == '__main__':
    args = [a for a in sys.argv[1:]]
    if '--regions' in args:
        rank_all_regions()
    elif '--all' in args:
        rank_all()
    else:
        cases = args or ['00493']
        cases = [c if c.startswith('BraTS2021_') else f'BraTS2021_{c}' for c in cases]
        for c in cases:
            diagnose(c)
