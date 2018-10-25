#! /bin/bash
rm out/*
rm err/*
rm log/*

#datasets name
#foreach VAR ("cs")
#no_cross_tune_grow_oracle
#foreach VAR ("dataset2" "dataset3")
#  foreach VAR1 ("accuracy" "recall" "precision" "false_alarm")
#    bsub -q standard -W 5000 -n 16 -o ./out/$VAR$VAR1.out.%J -e ./err/$VAR$VAR1.err.%J /share/tjmenzie/aagrawa8/miniconda2/bin/python2.7 main.py _test "$VAR" "$VA1" > log/"$VAR""$VAR1".log
#  end
#end

#source requirements.sh
#source activate tensorflow_gpu
for e in "untuned.py" "LDA.py" "LDADE.py";
do
for f in "cs_abinit" "cs_lammps" "cs_libmesh" "cs_mdanalysis";
do
    sbatch run.mpi $e $f
done;
done;
