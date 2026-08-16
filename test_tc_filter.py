"""Does removing WT components that contain no tumour-core prediction help?

Rationale from the 00493 diagnosis: peritumoral edema is by definition adjacent to
tumour tissue. A WT connected component holding no TC voxel anywhere is either a
rare pure-edema satellite or non-tumour FLAIR pathology (chronic white-matter
hyperintensity). Compare against plain largest-connected-component filtering.

No training, no GPU: operates on the predictions already on disk.
"""
import glob, os
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy import stats

PRED = ('/root/autodl-tmp/data/nnUNet_results/Dataset137_BraTS2021/'
        'nnUNetTrainer_FADSR_99epochs__nnUNetPlans__2d/fold_0/predictions_2d_test')
GT = '/root/autodl-tmp/data/nnUNet_raw/Dataset137_BraTS2021/labelsTs'
WT_IDS, TC_IDS = (1, 2, 3), (2, 3)


def dice(p, g):
    s = p.sum() + g.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(p, g).sum() / s


def filter_no_tc(pred, min_tc=1):
    """Drop WT components containing fewer than min_tc core voxels."""
    wt, tc = np.isin(pred, WT_IDS), np.isin(pred, TC_IDS)
    lab, n = ndimage.label(wt)
    if n <= 1:
        return wt.copy(), 0
    keep = np.zeros_like(wt)
    dropped = 0
    for i in range(1, n + 1):
        comp = lab == i
        if np.logical_and(comp, tc).sum() >= min_tc:
            keep |= comp
        else:
            dropped += 1
    # never return an empty mask -- fall back to the largest component
    if not keep.any():
        sizes = ndimage.sum(wt, lab, range(1, n + 1))
        keep = lab == (int(np.argmax(sizes)) + 1)
        dropped = n - 1
    return keep, dropped


def filter_lcc(pred):
    wt = np.isin(pred, WT_IDS)
    lab, n = ndimage.label(wt)
    if n <= 1:
        return wt.copy()
    sizes = ndimage.sum(wt, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def main():
    cases = sorted(os.path.basename(f).replace('.nii.gz', '')
                   for f in glob.glob(f'{PRED}/*.nii.gz'))
    rows = []
    for c in cases:
        pred = sitk.GetArrayFromImage(sitk.ReadImage(f'{PRED}/{c}.nii.gz')).astype(np.uint8)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(f'{GT}/{c}.nii.gz')).astype(np.uint8)
        g_wt = np.isin(gt, WT_IDS)
        raw = np.isin(pred, WT_IDS)
        f_tc, dropped = filter_no_tc(pred)
        rows.append(dict(case=c, base=dice(raw, g_wt), tcf=dice(f_tc, g_wt),
                         lcc=dice(filter_lcc(pred), g_wt), dropped=dropped,
                         drop_ml=float((raw.sum() - f_tc.sum())) / 1000.0))
        del pred, gt

    base = np.array([r['base'] for r in rows])
    tcf = np.array([r['tcf'] for r in rows])
    lcc = np.array([r['lcc'] for r in rows])

    print(f'n = {len(rows)} test cases\n')
    print(f'{"method":34s} {"mean WT":>8} {"delta":>8} {"better":>7} {"worse":>6} {"p":>10}')
    print(f'{"raw prediction":34s} {base.mean():8.4f} {"":>8} {"":>7} {"":>6}')
    for name, arr in (('largest-connected-component', lcc),
                      ('drop WT components with no TC', tcf)):
        d = arr - base
        w = stats.wilcoxon(arr, base) if np.any(d != 0) else None
        print(f'{name:34s} {arr.mean():8.4f} {d.mean():+8.4f} '
              f'{int((d > 1e-9).sum()):7d} {int((d < -1e-9).sum()):6d} '
              f'{w.pvalue:10.2e}' if w else f'{name:34s} {arr.mean():8.4f}')

    d = tcf - base
    sd = d.std(ddof=1)
    print(f'\nTC-filter paired sd={sd:.4f}  sem={sd/np.sqrt(len(d)):.5f}  '
          f'delta/sem = {d.mean()/(sd/np.sqrt(len(d))):.1f} sigma')
    print(f'cases where any component was dropped: '
          f'{sum(1 for r in rows if r["dropped"] > 0)}/{len(rows)}')

    print('\n--- worst 10 cases by raw WT dice ---')
    print(f'{"case":18} {"raw":>7} {"TCfilt":>7} {"delta":>7} {"LCC":>7} {"nDrop":>6} {"dropped mL":>10}')
    for r in sorted(rows, key=lambda x: x['base'])[:10]:
        print(f'{r["case"]:18} {r["base"]:7.3f} {r["tcf"]:7.3f} {r["tcf"]-r["base"]:+7.3f} '
              f'{r["lcc"]:7.3f} {r["dropped"]:6d} {r["drop_ml"]:10.1f}')

    hurt = sorted((r for r in rows if r['tcf'] < r['base'] - 1e-9),
                  key=lambda x: x['tcf'] - x['base'])[:8]
    if hurt:
        print('\n--- cases the TC filter HURTS most (pure-edema satellites?) ---')
        for r in hurt:
            print(f'{r["case"]:18} {r["base"]:7.3f} -> {r["tcf"]:7.3f} '
                  f'({r["tcf"]-r["base"]:+.3f})  dropped {r["drop_ml"]:.1f} mL')

    print('\nnote: TC and ET are untouched by this filter, so their Dice cannot change.')


if __name__ == '__main__':
    main()
