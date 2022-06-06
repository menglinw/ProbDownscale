#!/bin/bash
save_path=/scratch1/menglinw/Results/6_3_results
cur_path=`pwd`

for meta in meta nonmeta
do
  mkdir $save_path/$meta
  if [ "$meta" == "nonmeta" ]
  then
    for lat in 1 2 3 4
    do
      for lon in 1 2 3 4 5 6 7
      do
        mkdir $save_path/$meta/$lat$lon
        cp batch_run.sh $save_path/$meta/$lat$lon
        cd $save_path/$meta/$lat$lon
        echo "
python3 /scratch1/menglinw/ProbDownscale/probdownscale/test_script.py ${save_path} ${lat}${lon} nonbeta ${meta}">>batch_run.sh
        echo "${save_path} ${lat}${lon} nonbeta ${meta}"
        sbatch batch_run.sh
        cd $cur_path
      done
    done
  else
    for beta in beta nonbeta
    do
      mkdir $save_path/$meta/$beta
      for lat in 1 2 3 4
      do
        for lon in 1 2 3 4 5 6 7
        do
          mkdir $save_path/$meta/$beta/$lat$lon
          cp batch_run.sh $save_path/$meta/$beta/$lat$lon
          cd $save_path/$meta/$beta/$lat$lon
          echo "
python3 /scratch1/menglinw/ProbDownscale/probdownscale/test_script.py ${save_path} ${lat}${lon} ${beta} ${meta}">>batch_run.sh
          echo "${save_path} ${lat}${lon} ${beta} ${meta}"
          sbatch batch_run.sh
          cd $cur_path
        done
      done
    done
  fi
done