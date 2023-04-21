# 272-Nalazoo

## 0. Enviroments
* Python >= 3.7
* paddlepaddle >= 2.0.0
* CUDA >= 10.2(paddlepaddle 공식 documents에서는 10.2나 11.6을 권장함.)
## 1.Installation
[paddlepaddle install](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/install/pip/windows-pip_en.html)
(GPU: CUDA version에 맞게 설치(11.6 install시 11.7에서도 구동되는 것 확인))  
[model weights down](https://drive.google.com/file/d/1boNMCjPc3uTMe4YgQsAe_kMef0FXwMvH/view?usp=sharing)
(다운 후 압축해제 결과물 파일들을 모두 output directory로 이동.)

```
pip install -r requirements.txt --ignore-installed
```

## 2. Inference
```commandline
python tools/infer_e2e.py -c "configs/e2e/pet_receipts.yml" -o Global.infer_img=<Image path(directory or image)> Global.pretrained_model="./output/best_accuracy" Global.load_static_weights=False
```

## 3. Train
```commandline
python tools/train.py -c configs/e2e/pet_receipts.yml
```

## 4. 이미지 경로
dataset 폴더에 임의의 폴더명으로 저장. 이후 infer_e2e등의 파라미터로 폴더명 전달.