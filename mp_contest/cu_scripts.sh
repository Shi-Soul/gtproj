
for file in "cu_base.pt" "cu_lp5_3e7.pt" "cu_lp5_30e7.pt"; do
    # echo "Processing file: $file" 
    # CUDA_VISIBLE_DEVICES=0 python baselines/evaluation/eval_tess.py --num_episodes 100 --eval_on_scenario True --scenario clean_up --policies_dir /home/wjxie/wjxie/env/gtproj/tess/Tess/saved_models/clean_up/$file --expname $file >> cu_result/$file.cu0.log &
    CUDA_VISIBLE_DEVICES=1 python baselines/evaluation/eval_tess.py --num_episodes 100 --eval_on_scenario True --scenario clean_up_7 --policies_dir /home/wjxie/wjxie/env/gtproj/tess/Tess/saved_models/clean_up/$file --expname $file >> cu_result/$file.cu7_2.log &
done

# CUDA_VISIBLE_DEVICES=0 python baselines/evaluation/eval_tess.py --num_episodes 20 --eval_on_scenario True --scenario clean_up --policies_dir /home/wjxie/wjxie/env/gtproj/tess/Tess/saved_models/clean_up/cu_base.pt >> cu_result/cu_base.pt.cu0.log &
# CUDA_VISIBLE_DEVICES=1 python baselines/evaluation/eval_tess.py --num_episodes 20 --eval_on_scenario True --scenario clean_up_7 --policies_dir /home/wjxie/wjxie/env/gtproj/tess/Tess/saved_models/clean_up/cu_base.pt >> cu_result/cu_base.pt.cu7.log &
# CUDA_VISIBLE_DEVICES=1 python baselines/evaluation/eval_tess.py --num_episodes 20 --eval_on_scenario True --scenario clean_up --policies_dir /home/wjxie/wjxie/env/gtproj/tess/Tess/saved_models/clean_up/cu_lp4.pt >> cu_result/cu_lp4.pt.cu0.log &
# CUDA_VISIBLE_DEVICES=0 python baselines/evaluation/eval_tess.py --num_episodes 20 --eval_on_scenario True --scenario clean_up_7 --policies_dir /home/wjxie/wjxie/env/gtproj/tess/Tess/saved_models/clean_up/cu_lp4.pt >> cu_result/cu_lp4.pt.cu7.log &
# CUDA_VISIBLE_DEVICES=1 python baselines/evaluation/eval_tess.py --num_episodes 20 --eval_on_scenario True --scenario clean_up --policies_dir /home/wjxie/wjxie/env/gtproj/tess/Tess/saved_models/clean_up/cu_lp2_6e7.pt >> cu_result/cu_lp2_6e7.pt.cu0.log &
# CUDA_VISIBLE_DEVICES=1 python baselines/evaluation/eval_tess.py --num_episodes 20 --eval_on_scenario True --scenario clean_up_7 --policies_dir /home/wjxie/wjxie/env/gtproj/tess/Tess/saved_models/clean_up/cu_lp2_6e7.pt >> cu_result/cu_lp2_6e7.pt.cu7.log &
# CUDA_VISIBLE_DEVICES=2 python baselines/evaluation/eval_tess.py --num_episodes 20 --eval_on_scenario True --scenario clean_up --policies_dir /home/wjxie/wjxie/env/gtproj/tess/Tess/saved_models/clean_up/cu_lp5_3e7.pt >> cu_result/cu_lp5_3e7.pt.cu0.log &
# CUDA_VISIBLE_DEVICES=1 python baselines/evaluation/eval_tess.py --num_episodes 20 --eval_on_scenario True --scenario clean_up_7 --policies_dir /home/wjxie/wjxie/env/gtproj/tess/Tess/saved_models/clean_up/cu_lp5_3e7.pt >> cu_result/cu_lp5_3e7.pt.cu7.log &
# CUDA_VISIBLE_DEVICES=0 python baselines/evaluation/eval_tess.py --num_episodes 20 --eval_on_scenario True --scenario clean_up --policies_dir /home/wjxie/wjxie/env/gtproj/tess/Tess/saved_models/clean_up/cu_lp5_30e7.pt >> cu_result/cu_lp5_30e7.pt.cu0.log &
# CUDA_VISIBLE_DEVICES=1 python baselines/evaluation/eval_tess.py --num_episodes 20 --eval_on_scenario True --scenario clean_up_7 --policies_dir /home/wjxie/wjxie/env/gtproj/tess/Tess/saved_models/clean_up/cu_lp5_30e7.pt >> cu_result/cu_lp5_30e7.pt.cu7.log &

CUDA_VISIBLE_DEVICES=1 python baselines/evaluation/eval_tess.py --num_episodes 5 --eval_on_scenario True --scenario clean_up_7 --policies_dir /home/wjxie/wjxie/env/gtproj/tess/Tess/saved_models/clean_up/cu_lp5_30e7.pt --create_videos True --video_dir /home/wjxie/wjxie/env/gtproj/mp_contest/video_lp5_cu7

CUDA_VISIBLE_DEVICES=0 python baselines/evaluation/eval_tess.py --num_episodes 5 --eval_on_scenario True --scenario clean_up --policies_dir /home/wjxie/wjxie/env/gtproj/tess/Tess/saved_models/clean_up/cu_lp5_30e7.pt --create_videos True --video_dir /home/wjxie/wjxie/env/gtproj/mp_contest/video_lp5_cu0


CUDA_VISIBLE_DEVICES=1 python baselines/evaluation/eval_tess.py --num_episodes 5 --eval_on_scenario True --scenario clean_up_7 --policies_dir /home/wjxie/wjxie/env/gtproj/tess/Tess/saved_models/clean_up/cu_base.pt --create_videos True --video_dir /home/wjxie/wjxie/env/gtproj/mp_contest/video_base_cu7 &

CUDA_VISIBLE_DEVICES=2 python baselines/evaluation/eval_tess.py --num_episodes 5 --eval_on_scenario True --scenario clean_up --policies_dir /home/wjxie/wjxie/env/gtproj/tess/Tess/saved_models/clean_up/cu_base.pt --create_videos True --video_dir /home/wjxie/wjxie/env/gtproj/mp_contest/video_base_cu0 &

