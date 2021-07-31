#! /bin/sh

#a=${1-0}

#if (( $a == 0 || $a == 1 )); then 
echo "run compare.py" 
python3 compare.py
#fi

echo ""

#if (( $a == 0 || $a == 2 )); then 
echo "run plot_best.py" 
python3 plot_bests.py
#fi

echo ""

#if (( $a == 0 || $a == 3 )); then 
echo "run compare_time.py" 
#python3 compare_time.py
#fi

echo ""

echo "change directory to 'results'"
cd results

echo "trim images"
for file in ./*.png;
	do echo $file; 
	convert $file -trim $file;
done

echo "copy *.png files."
cp *.png ~/Documents/MLDM/S2/Internship/OT/report_internship_labHC/images/
