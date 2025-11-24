
# ============================================================
# Phase 2: Loss Function 실험
# ============================================================

echo "=========================================="
echo "Phase 2: Loss Function 실험"
echo "=========================================="

# 2-1. ArcFace Loss (scale=64, margin=0.5) - Train1
echo "실험 2-1: ArcFace Loss (scale=64, margin=0.5, train1)"
python code/trainEmbedNet.py \
    --gpu 0 \
    --train_path /mnt/home/ee40034/data/train1 \
    --test_path /mnt/home/ee40034/data/val \
    --test_list /mnt/home/ee40034/data/val_pairs.csv \
    --save_path ./exps/phase2_arcface_s64_m05_train1 \
    --model ResNet18 \
    --trainfunc arcface \
    --optimizer adam \
    --scheduler steplr \
    --lr 0.001 \
    --lr_decay 0.90 \
    --scale 64 \
    --margin 0.5 \
    --nClasses 9500 \
    --max_epoch 10

# 2-2. ArcFace Loss (scale=64, margin=0.5) - Train2
echo "실험 2-2: ArcFace Loss (scale=64, margin=0.5, train2)"
python code/trainEmbedNet.py \
    --gpu 0 \
    --train_path /mnt/home/ee40034/data/train2 \
    --test_path /mnt/home/ee40034/data/val \
    --test_list /mnt/home/ee40034/data/val_pairs.csv \
    --save_path ./exps/phase2_arcface_s64_m05_train2 \
    --model ResNet18 \
    --trainfunc arcface \
    --optimizer adam \
    --scheduler steplr \
    --lr 0.001 \
    --lr_decay 0.90 \
    --scale 64 \
    --margin 0.5 \
    --nClasses 949 \
    --max_epoch 10

# 2-3. ArcFace Loss (scale=64, margin=0.3) - Train1
echo "실험 2-3: ArcFace Loss (scale=64, margin=0.3, train1)"
python code/trainEmbedNet.py \
    --gpu 0 \
    --train_path /mnt/home/ee40034/data/train1 \
    --test_path /mnt/home/ee40034/data/val \
    --test_list /mnt/home/ee40034/data/val_pairs.csv \
    --save_path ./exps/phase2_arcface_s64_m03_train1 \
    --model ResNet18 \
    --trainfunc arcface \
    --optimizer adam \
    --scheduler steplr \
    --lr 0.001 \
    --lr_decay 0.90 \
    --scale 64 \
    --margin 0.3 \
    --nClasses 9500 \
    --max_epoch 10

# 2-4. ArcFace Loss (scale=30, margin=0.5) - Train1
echo "실험 2-4: ArcFace Loss (scale=30, margin=0.5, train1)"
python code/trainEmbedNet.py \
    --gpu 0 \
    --train_path /mnt/home/ee40034/data/train1 \
    --test_path /mnt/home/ee40034/data/val \
    --test_list /mnt/home/ee40034/data/val_pairs.csv \
    --save_path ./exps/phase2_arcface_s30_m05_train1 \
    --model ResNet18 \
    --trainfunc arcface \
    --optimizer adam \
    --scheduler steplr \
    --lr 0.001 \
    --lr_decay 0.90 \
    --scale 30 \
    --margin 0.5 \
    --nClasses 9500 \
    --max_epoch 10

# 2-5. ArcFace Loss (scale=30, margin=0.3) - Train1
echo "실험 2-5: ArcFace Loss (scale=30, margin=0.3, train1)"
python code/trainEmbedNet.py \
    --gpu 0 \
    --train_path /mnt/home/ee40034/data/train1 \
    --test_path /mnt/home/ee40034/data/val \
    --test_list /mnt/home/ee40034/data/val_pairs.csv \
    --save_path ./exps/phase2_arcface_s30_m03_train1 \
    --model ResNet18 \
    --trainfunc arcface \
    --optimizer adam \
    --scheduler steplr \
    --lr 0.001 \
    --lr_decay 0.90 \
    --scale 30 \
    --margin 0.3 \
    --nClasses 9500 \
    --max_epoch 10

# 2-6. Triplet Loss (Hard Negative Mining, margin=0.1) - Train1
echo "실험 2-6: Triplet Hard Loss (margin=0.1, train1)"
python code/trainEmbedNet.py \
    --gpu 0 \
    --train_path /mnt/home/ee40034/data/train1 \
    --test_path /mnt/home/ee40034/data/val \
    --test_list /mnt/home/ee40034/data/val_pairs.csv \
    --save_path ./exps/phase2_triplet_hard_m01_train1 \
    --model ResNet18 \
    --trainfunc triplet_hard \
    --optimizer adam \
    --scheduler steplr \
    --lr 0.001 \
    --lr_decay 0.90 \
    --margin 0.1 \
    --nPerClass 2 \
    --max_epoch 10

# 2-7. Triplet Loss (Hard Negative Mining, margin=0.2) - Train1
echo "실험 2-7: Triplet Hard Loss (margin=0.2, train1)"
python code/trainEmbedNet.py \
    --gpu 0 \
    --train_path /mnt/home/ee40034/data/train1 \
    --test_path /mnt/home/ee40034/data/val \
    --test_list /mnt/home/ee40034/data/val_pairs.csv \
    --save_path ./exps/phase2_triplet_hard_m02_train1 \
    --model ResNet18 \
    --trainfunc triplet_hard \
    --optimizer adam \
    --scheduler steplr \
    --lr 0.001 \
    --lr_decay 0.90 \
    --margin 0.2 \
    --nPerClass 2 \
    --max_epoch 10

# 2-8. Triplet Loss (Hard Negative Mining, margin=0.3) - Train1
echo "실험 2-8: Triplet Hard Loss (margin=0.3, train1)"
python code/trainEmbedNet.py \
    --gpu 0 \
    --train_path /mnt/home/ee40034/data/train1 \
    --test_path /mnt/home/ee40034/data/val \
    --test_list /mnt/home/ee40034/data/val_pairs.csv \
    --save_path ./exps/phase2_triplet_hard_m03_train1 \
    --model ResNet18 \
    --trainfunc triplet_hard \
    --optimizer adam \
    --scheduler steplr \
    --lr 0.001 \
    --lr_decay 0.90 \
    --margin 0.3 \
    --nPerClass 2 \
    --max_epoch 10

# 2-9. Softmax + ArcFace Combined (weight 0.7:0.3) - Train1
echo "실험 2-9: Softmax + ArcFace Combined (weight 0.7:0.3, train1)"
python code/trainEmbedNet.py \
    --gpu 0 \
    --train_path /mnt/home/ee40034/data/train1 \
    --test_path /mnt/home/ee40034/data/val \
    --test_list /mnt/home/ee40034/data/val_pairs.csv \
    --save_path ./exps/phase2_combined_07_03_train1 \
    --model ResNet18 \
    --trainfunc softmax_arcface \
    --optimizer adam \
    --scheduler steplr \
    --lr 0.001 \
    --lr_decay 0.90 \
    --scale 64 \
    --margin 0.5 \
    --weight_softmax 0.7 \
    --weight_arcface 0.3 \
    --nClasses 9500 \
    --max_epoch 10

# 2-10. Softmax + ArcFace Combined (weight 0.5:0.5) - Train1
echo "실험 2-10: Softmax + ArcFace Combined (weight 0.5:0.5, train1)"
python code/trainEmbedNet.py \
    --gpu 0 \
    --train_path /mnt/home/ee40034/data/train1 \
    --test_path /mnt/home/ee40034/data/val \
    --test_list /mnt/home/ee40034/data/val_pairs.csv \
    --save_path ./exps/phase2_combined_05_05_train1 \
    --model ResNet18 \
    --trainfunc softmax_arcface \
    --optimizer adam \
    --scheduler steplr \
    --lr 0.001 \
    --lr_decay 0.90 \
    --scale 64 \
    --margin 0.5 \
    --weight_softmax 0.5 \
    --weight_arcface 0.5 \
    --nClasses 9500 \
    --max_epoch 10

echo "=========================================="
echo "Phase 1 & Phase 2 실험 완료!"
echo "=========================================="

