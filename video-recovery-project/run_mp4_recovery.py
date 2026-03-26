from pathlib import Path
import json
import subprocess

import imageio_ffmpeg


def probe_ok(ffmpeg: str, path: Path) -> bool:
    cmd = [ffmpeg, '-v', 'error', '-i', str(path), '-f', 'null', '-']
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    err = (p.stderr or '').lower()
    fatal_markers = [
        'error opening input',
        'invalid data found when processing input',
        'no such file or directory',
    ]
    return not any(m in err for m in fatal_markers)


def to_mp4(ffmpeg: str, inp: Path, out: Path) -> tuple[int, str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        '-y',
        '-err_detect',
        'ignore_err',
        '-fflags',
        '+genpts+igndts',
        '-i',
        str(inp),
        '-map',
        '0:v:0?',
        '-map',
        '0:a:0?',
        '-c:v',
        'libx264',
        '-preset',
        'veryfast',
        '-crf',
        '22',
        '-c:a',
        'aac',
        '-b:a',
        '160k',
        str(out),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, '\n'.join((p.stderr or '').splitlines()[-12:])


def main() -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    root = Path('/Users/matthewconway/Documents/GitHub/Comp-Sim-Numerical-Methods/video-recovery-project')
    source_dir = Path('/Users/matthewconway/Desktop/currupt_vids')
    recovered_dir = root / 'recovered_output'
    final_dir = root / 'recovered_mp4'
    report_dir = root / 'reports'

    final_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted(source_dir.glob('*.AVI')) + sorted(source_dir.glob('*.avi'))
    seen = set()
    unique_source = []
    for p in source_files:
        if p.name not in seen:
            seen.add(p.name)
            unique_source.append(p)

    skip_order = [0, 512, 1024, 2048, 4096, 8192]

    results = []
    for src in unique_source:
        stem = src.stem
        out_mp4 = final_dir / f'{stem}.mp4'

        candidates = []
        mkv = recovered_dir / f'{stem}.copyfix.mkv'
        avi = recovered_dir / f'{stem}.copyfix.avi'
        if mkv.exists():
            candidates.append(('copyfix_mkv', mkv))
        if avi.exists():
            candidates.append(('copyfix_avi', avi))

        for s in skip_order:
            h = recovered_dir / f'{stem}.hdrfix.t1.s{s}.avi'
            if h.exists():
                candidates.append((f'hdrfix_s{s}', h))

        recovered = False
        method = ''
        detail = ''

        for name, cand in candidates:
            if not probe_ok(ffmpeg, cand):
                continue
            rc, tail = to_mp4(ffmpeg, cand, out_mp4)
            if rc == 0 and out_mp4.exists() and out_mp4.stat().st_size > 0:
                recovered = True
                method = name
                detail = tail
                break

        results.append(
            {
                'source': str(src),
                'output_mp4': str(out_mp4),
                'recovered_mp4': recovered,
                'method': method,
                'detail_tail': detail,
            }
        )

    ok = sum(1 for r in results if r['recovered_mp4'])
    fail = len(results) - ok
    print(f'Total source files: {len(results)}')
    print(f'MP4 recovered     : {ok}')
    print(f'MP4 failed        : {fail}')

    report_path = report_dir / 'mp4_recovery_report.json'
    with report_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f'Report written: {report_path}')

    for p in sorted(final_dir.glob('*.mp4')):
        print(p.name)


if __name__ == '__main__':
    main()
