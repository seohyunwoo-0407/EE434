#!/bin/bash
# Phase 1 & Phase 2 실험 실행 스크립트
# 모든 실험을 순차적으로 실행합니다

# ============================================================
# Phase 1: Baseline 확인
# ============================================================

echo "=========================================="
echo "Phase 1: Baseline 확인"
echo "=========================================="

# 1-1. Baseline (Softmax) - Train1
echo "실험 1-1: Baseline Softmax (train1)"
python code/trainEmbedNet.py \
    --gpu 0 \
    --train_path /mnt/home/ee40034/data/train1 \
    --test_path /mnt/home/ee40034/data/val \
    --test_list /mnt/home/ee40034/data/val_pairs.csv \
    --save_path ./exps/phase1_baseline_softmax_train1 \
    --model ResNet18 \
    --trainfunc softmax \
    --optimizer adam \
    --scheduler steplr \
    --lr 0.001 \
    --lr_decay 0.90 \
    --nClasses 9500 \
    --max_epoch 12

# 1-2. Baseline (Softmax) - Train2
echo "실험 1-2: Baseline Softmax (train2)"
python code/trainEmbedNet.py \
    --gpu 0 \
    --train_path /mnt/home/ee40034/data/train2 \
    --test_path /mnt/home/ee40034/data/val \
    --test_list /mnt/home/ee40034/data/val_pairs.csv \
    --save_path ./exps/phase1_baseline_softmax_train2 \
    --model ResNet18 \
    --trainfunc softmax \
    --optimizer adam \
    --scheduler steplr \
    --lr 0.001 \
    --lr_decay 0.90 \
    --nClasses 949 \
    --max_epoch 12

# 1-3. Baseline (Triplet Loss, random) - Train1
echo "실험 1-3: Baseline Triplet Loss (train1)"
python code/trainEmbedNet.py \
    --gpu 0 \
    --train_path /mnt/home/ee40034/data/train1 \
    --test_path /mnt/home/ee40034/data/val \
    --test_list /mnt/home/ee40034/data/val_pairs.csv \
    --save_path ./exps/phase1_baseline_triplet_train1 \
    --model ResNet18 \
    --trainfunc triplet \
    --optimizer adam \
    --scheduler steplr \
    --lr 0.001 \
    --lr_decay 0.90 \
    --margin 0.1 \
    --nPerClass 2 \
    --max_epoch 12

# 1-4. Baseline (Triplet Loss, random) - Train2
echo "실험 1-4: Baseline Triplet Loss (train2)"
python code/trainEmbedNet.py \
    --gpu 0 \
    --train_path /mnt/home/ee40034/data/train2 \
    --test_path /mnt/home/ee40034/data/val \
    --test_list /mnt/home/ee40034/data/val_pairs.csv \
    --save_path ./exps/phase1_baseline_triplet_train2 \
    --model ResNet18 \
    --trainfunc triplet \
    --optimizer adam \
    --scheduler steplr \
    --lr 0.001 \
    --lr_decay 0.90 \
    --margin 0.1 \
    --nPerClass 2 \
    --max_epoch 12

# 1-5. Baseline (Softmax) - Train1+Train2 합쳐서 학습
echo "실험 1-5: Baseline Softmax (train1+train2 합쳐서)"
# Train1과 Train2를 합친 임시 디렉토리 생성
TRAIN_COMBINED_DIR="./data_combined_train1_train2"
if [ ! -d "$TRAIN_COMBINED_DIR" ]; then
    echo "  Train1과 Train2를 합친 디렉토리 생성 중..."
    mkdir -p "$TRAIN_COMBINED_DIR"
    # Train1의 모든 identity 디렉토리를 심볼릭 링크로 복사
    for dir in /mnt/home/ee40034/data/train1/*/; do
        identity=$(basename "$dir")
        ln -s "$dir" "$TRAIN_COMBINED_DIR/$identity" 2>/dev/null || true
    done
    # Train2의 모든 identity 디렉토리를 심볼릭 링크로 복사 (중복 시 train2 우선)
    for dir in /mnt/home/ee40034/data/train2/*/; do
        identity=$(basename "$dir")
        if [ ! -e "$TRAIN_COMBINED_DIR/$identity" ]; then
            ln -s "$dir" "$TRAIN_COMBINED_DIR/$identity" 2>/dev/null || true
        fi
    done
    echo "  디렉토리 생성 완료"
fi

python code/trainEmbedNet.py \
    --gpu 0 \
    --train_path "$TRAIN_COMBINED_DIR" \
    --test_path /mnt/home/ee40034/data/val \
    --test_list /mnt/home/ee40034/data/val_pairs.csv \
    --save_path ./exps/phase1_baseline_softmax_train1_train2 \
    --model ResNet18 \
    --trainfunc softmax \
    --optimizer adam \
    --scheduler steplr \
    --lr 0.001 \
    --lr_decay 0.90 \
    --nClasses 3831 \
    --max_epoch 12

# 1-6. Baseline (Triplet Loss) - Train1+Train2 합쳐서 학습
echo "실험 1-6: Baseline Triplet Loss (train1+train2 합쳐서)"
python code/trainEmbedNet.py \
    --gpu 0 \
    --train_path "$TRAIN_COMBINED_DIR" \
    --test_path /mnt/home/ee40034/data/val \
    --test_list /mnt/home/ee40034/data/val_pairs.csv \
    --save_path ./exps/phase1_baseline_triplet_train1_train2 \
    --model ResNet18 \
    --trainfunc triplet \
    --optimizer adam \
    --scheduler steplr \
    --lr 0.001 \
    --lr_decay 0.90 \
    --margin 0.1 \
    --nPerClass 2 \
    --max_epoch 12
