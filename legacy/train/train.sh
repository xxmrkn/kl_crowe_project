PROJECT='/win/salmon/user/masuda/project/vit_kl_crowe'
NUM_GPU=1
NUM_CORE=4

NODE='cl-dragonet'
#CONFIG='config/adaptive_sigma/1fold.json'
sbatch --gres=gpu:${NUM_GPU} -n ${NUM_CORE} -D ${PROJECT} -w ${NODE} -o slurm/slurm_%j.out\
  --wrap="singularity exec --nv -B /win/salmon/user/masuda /win/salmon/user/hakotani/for/masuda/masuda_cuda11.3.1_pytorch1.11.0.sif\
  python3 demo.py"

#sbatch --gres=gpu:1 -c 2 -w cl-panda -D $(pwd) -o ./slurm/slurm-%j_tmp_all.out\
#  --wrap="singularity exec --nv -B /win/salmon/user/masuda /win/salmon/user/mazen/Cluster/py35/AIO.simg\
#  python3 test.py"