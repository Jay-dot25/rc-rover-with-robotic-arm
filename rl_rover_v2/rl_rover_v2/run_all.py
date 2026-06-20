"""
run_all.py — One command to train + simulate
=============================================
Usage:
    python run_all.py               # train then launch live simulation
    python run_all.py --train-only  # only train
    python run_all.py --sim-only    # only simulate (needs trained model)
    python run_all.py --video       # train + save MP4
"""

import argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

parser = argparse.ArgumentParser()
parser.add_argument("--train-only", action="store_true")
parser.add_argument("--sim-only",   action="store_true")
parser.add_argument("--video",      action="store_true")
parser.add_argument("--difficulty", default="medium", choices=["easy","medium","hard"])
args = parser.parse_args()

if not args.sim_only:
    print("\n[1/2] Training ...\n")
    from train import train
    train()

if not args.train_only:
    print("\n[2/2] Launching simulation ...\n")
    sim_args = []
    if args.video:
        sim_args += ["--video"]
    sim_args += ["--difficulty", args.difficulty]
    sys.argv = ["simulate.py"] + sim_args
    from simulate import main
    main()
