#!/usr/bin/env python3
"""Run fixed SAM or nnInteractive partitions on GPUs with enough free memory."""
import argparse
from collections import deque
import fcntl
import os
from pathlib import Path
import subprocess
import time


def free_memory():
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'],
        text=True,
    )
    return {int(index): (int(free), uuid.strip())
            for index, uuid, free in (line.split(',') for line in output.splitlines())}


def run(args):
    gpu_ids = tuple(map(int, args.gpus.split(',')))
    leases = Path('/tmp') / f'brats_foundation_sam_gpu_{os.getuid()}'
    leases.mkdir(mode=0o700, exist_ok=True)
    pending = deque(range(args.workers))
    running, failed, retries, cooldown = {}, [], {}, {}
    while pending or running:
        for gpu, (index, worker, log, start) in list(running.items()):
            status = worker.poll()
            if status is None:
                continue
            del running[gpu]
            if status == 0:
                continue
            with log.open() as stream:
                stream.seek(start)
                error = stream.read()
            memory_error = any(message in error for message in ('Need 8 GiB free', 'Insufficient free CUDA memory',
                                                                'CUDA out of memory'))
            attempts = retries.get(index, 0)
            if memory_error and attempts < len(gpu_ids):
                retries[index] = attempts + 1
                cooldown[gpu] = time.monotonic() + 45
                pending.append(index)
                print(f'GPU {gpu} reached its memory limit. Requeued partition {index}.', flush=True)
            else:
                failed.append((index, status, log))

        if pending:
            try:
                memory = free_memory()
            except subprocess.CalledProcessError:
                time.sleep(10)
                continue
            missing = set(gpu_ids) - memory.keys()
            if missing:
                raise ValueError(f'Unknown physical GPU IDs: {sorted(missing)}')
            for gpu in sorted(gpu_ids, key=lambda value: -memory[value][0]):
                if not pending:
                    break
                free, uuid = memory[gpu]
                if gpu in running or time.monotonic() < cooldown.get(gpu, 0) or free < args.min_mib:
                    continue
                with (leases / uuid).open('a') as lease:
                    try:
                        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    index = pending.popleft()
                    log = Path(f'{args.log_prefix}_{index}.log')
                    with log.open('a') as stream:
                        start = stream.tell()
                        print(f'Physical GPU {gpu}, free {free} MiB', file=stream, flush=True)
                        # The worker owns the lease even if its dispatcher exits early.
                        task = subprocess.Popen(
                            [*args.command, args.index_argument, str(index), '--device', 'cuda:0'],
                            stdout=stream, stderr=subprocess.STDOUT, pass_fds=(lease.fileno(),),
                            env={**os.environ, 'CUDA_VISIBLE_DEVICES': uuid},
                        )
                running[gpu] = index, task, log, start

        if pending or running:
            time.sleep(10)

    for index, status, path in failed:
        print(f'Partition {index} exited with code {status}. See {path}.', flush=True)
    return 1 if failed else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', required=True)
    p.add_argument('--workers', type=int, required=True)
    p.add_argument('--min-mib', type=int, required=True)
    p.add_argument('--index-argument', choices=('--shard-index', '--head'), required=True)
    p.add_argument('--log-prefix', required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    if args.command and args.command[0] == '--':
        args.command = args.command[1:]
    if not args.command or args.workers < 1:
        p.error('Expected a worker command and at least one partition')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
