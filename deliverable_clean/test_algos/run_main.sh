#! /bin/bash

echo "STARTING - run_main_ENT.sh"&
bash run_main_ENT.sh &

echo ""&

echo "STARTING - run_main_KBCM.sh"&
bash run_main_KBCM.sh &

echo ""&

echo "STARTING - run_main_TLp.sh"&
bash run_main_TLp.sh &

echo ""&

echo "STARTING - run_main_DBS.sh"&
bash run_main_DBS.sh &
