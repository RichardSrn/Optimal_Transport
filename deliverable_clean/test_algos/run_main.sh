#! /bin/bash

echo "STARTING - run_main_ENT.sh"&
sbatch run_main_ENT.sh &

echo ""&

echo "STARTING - run_main_KBCM.sh"&
sbatch run_main_KBCM.sh &

echo ""&

echo "STARTING - run_main_TLp.sh"&
sbatch run_main_TLp.sh &

echo ""&

echo "STARTING - run_main_DBS.sh"&
sbatch run_main_DBS.sh &