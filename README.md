# Road Illumination Classification Ensemble

## Цель работы

Классификация изображений дорожной сцены (день/ночь) по степени освещённости.  
Используется ансамбль моделей MobileNetV3, обученных по K‑Fold стратифицированной кросс-валидации.

---

## Установка

1. Клонируйте данный репозиторий
2. Создайте и активируйте виртуальное окружение:
```bash
python3 -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```
3.Установите зависимости:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Подготовка данных
Тестовые изображения разместите в папке data в формате:
```
data/
├──train/
│ ├──day/
| |
| └──night/
|
└──val/
  ├──day/
  |
  └──night/
```
Train/val датасеты доступны по ссылке: https://drive.google.com/drive/folders/1um4jBZCHfTK-myOY70m88hCiI20rDF43?usp=drive_link

## Обучение
Запустите обучение ансамбля моделей:
```bash
python src/train.py
```
По умолчанию используется 5‑фолдовая кросс-валидация.
Лучшие модели (по F1) сохраняются в папке checkpoints/.

## Инференс
Скрипт main.py позволяет выполнить предсказания по изображениями в указанной директории:
```bash
python src/main.py <папка_с_изображениями>
```
<папка_с_изображениями> — путь до тестовых изображений, например data/test

Пример вывода в консоль:
```
data/test\day8.png             → DAY   (p_day=0.97)
data/test\day9.png             → DAY   (p_day=0.87)
data/test\night10.png          → NIGHT (p_day=0.07)
data/test\night8.png           → NIGHT (p_day=0.46)
data/test\night9.png           → NIGHT (p_day=0.00)
```

## Поддержка Git LFS

Весовые файлы модели (`*.pt`) хранятся через Git LFS:

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git commit -m "Enable LFS for model weights"
git push origin main
```

## Настройка
Гиперпараметры обучения (NUM_FOLDS, EPOCHS, BATCH_SIZE, LR) можно изменить в src/train.py.

Biased-кроп настраивается через параметры alpha, beta внутри класса BiasedRandomResizedCrop.
