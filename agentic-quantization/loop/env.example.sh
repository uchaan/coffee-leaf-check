# Copy to env.sh and edit. Every script in this folder sources env.sh; the Python scripts read the same variables
# (env.sh is expected; unset variables fall back to ./llama.cpp and ./work relative to the current directory, the rest under $WORK).
export LLAMA_CPP_DIR=$HOME/llama.cpp            # llama.cpp checkout 9a7570587ce908b0073a0458877205b80627f393, built with CUDA (build/bin)
export WORK=$HOME/hn04b-work                     # models/, gguf/, calib/, logs/, runs/ live here
export PY=python                                 # python 3.11 with ../requirements.txt installed
export BRACOL_ROOT=$WORK/bracol/raw              # BRACOL leaf set: folder with dataset.csv and images/ (benchmark/README.md)
export HN04B_CACHE=$WORK/bracol/cache512         # preprocessed 512 px JPEGs (written by the harness)
export HN04B_RUNS=$WORK/runs                     # harness output root
export WIKITEXT=$WORK/calib/wikitext2_test.txt   # wikitext-2 raw test text, for the text-KLD check
export CALIB_TEXT=$WORK/calib/general_fwedu_c4.txt  # text calibration (FineWeb-Edu + C4) used by every build
export JMUBEN_ROOT=$WORK/jmuben/raw              # extracted JMuBEN + JMuBEN2 archives (optional second test set)
export HN04B_PUBLIC=$WORK/public                 # public GGUFs as <repo>/<file> (benchmark/fetch_public.py)
# export GPU_FREE_MB=30000                       # worker.sh waits until the GPU's used memory is below this (MB)
# export HN04B_SERVER_EXTRA="--cache-ram 0"      # extra llama-server flags for GPU runs (the example run used --cache-ram 0 from 10:12 KST)
# export HN04B_FILES=... HN04B_FINAL=...         # default: ../selection/files.csv and ../selection/final.csv
# export HN04B_HF_REPO=OWNER/REPO                # model repo that benchmark/fetch_ours.py downloads from
