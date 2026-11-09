#!/usr/bin/env python
"""Merge per-brick gaap-brick outputs into one catalogue, with per-brick blank-sky error scales.

    python examples/merge_bricks.py OUTDIR merged.fits [--tag TAG] [--all-rows]

HDU GAAP: the rows of every OUTDIR/<bbb>/gaap-<brick>[-<tag>].fits, brick-primary rows only (unless
--all-rows). HDU ERRSCALE: per brick with a blank-sky table, errscale_<b> and depth5_<b> (one value
per aperture; legacygaap.sky), to multiply the errors of that brick's rows by.
"""
import argparse
import glob
import os
import re
import numpy as np
import fitsio
from legacygaap import sky


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('outdir')
    p.add_argument('output')
    p.add_argument('--tag', default=None)
    p.add_argument('--all-rows', action='store_true', help='keep rows that are not brick_primary')
    a = p.parse_args()
    t = f'-{a.tag}' if a.tag else ''
    pat = re.compile(rf'^gaap-(\d{{4}}[pm]\d{{3}}){re.escape(t)}\.fits$')
    files = sorted(f for f in glob.glob(os.path.join(a.outdir, '*', f'gaap-*{t}.fits')) if pat.match(os.path.basename(f)))
    if not files:
        raise SystemExit(f'no gaap-<brick>{t}.fits under {a.outdir}/<bbb>/')
    parts, scales, names, bands = [], [], None, None
    for fn in files:
        brick = pat.match(os.path.basename(fn)).group(1)
        h = fitsio.read_header(fn, ext='GAAP')
        nm = [str(h[f'AP{k}']).strip() for k in range(h['NAP'])]
        bb = str(h['BANDS']).strip().split(',')
        if names is None:
            names, bands = nm, bb
        elif nm != names or bb != bands:
            raise SystemExit(f'{fn}: apertures {nm} / bands {bb} differ from {names} / {bands}')
        T = fitsio.read(fn, ext='GAAP')
        if not a.all_rows and 'brick_primary' in T.dtype.names:
            T = T[T['brick_primary']]
        parts.append(T)
        fs = os.path.join(os.path.dirname(fn), f'gaap-sky-{brick}{t}.fits')
        if os.path.exists(fs):
            S = fitsio.read(fs, ext='GAAP')
            scales.append((brick, {b: (sky.error_scale(S, b), sky.depth(S, b)) for b in bands}))
    T = np.concatenate(parts)
    hdr = [dict(name='NBRICK', value=len(files)), dict(name='NAP', value=len(names)), dict(name='BANDS', value=','.join(bands))]
    hdr += [dict(name=f'AP{k}', value=n) for k, n in enumerate(names)]
    tmp = a.output + '.tmp'
    fitsio.write(tmp, T, extname='GAAP', header=hdr, clobber=True)
    if scales:
        E = np.zeros(len(scales), dtype=[('brickname', 'S8')] + [(f'{q}_{b}', 'f4', (len(names),)) for b in bands
                                                                   for q in ('errscale', 'depth5')])
        for i, (brick, d) in enumerate(scales):
            E['brickname'][i] = brick
            for b, (s, dep) in d.items():
                E[f'errscale_{b}'][i], E[f'depth5_{b}'][i] = s, dep
        fitsio.write(tmp, E, extname='ERRSCALE', header=hdr)
    os.replace(tmp, a.output)
    print(f'{a.output}: {len(T):,} rows from {len(files)} bricks; error scales for {len(scales)} bricks')


if __name__ == '__main__':
    main()
