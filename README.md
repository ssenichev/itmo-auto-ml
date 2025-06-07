# AutoML Pipeline

## Запуск приложения через Docker

1. Соберите Docker-образ:

```sh
docker build -t automl-app .
```

2. Запустите контейнер:

```sh
docker run -p 7861:7861 automl-app
```

3. Откройте браузер и перейдите по адресу:

```
http://localhost:7861
```

Или запустите код напрямую:

1. Создайте и активируйте виртуальное окружение:
```sh
python -m venv venv
source venv/bin/activate
```

2. Скачайте `requirements.txt`:
```sh
pip install -r requirements.txt
```

3. Запустите `app.py`:
```sh
python3 app.py
```


## Описание
- Приложение предназначено для автоматизированного анализа табличных данных и построения моделей машинного обучения с использованием Spark, CatBoost и Gradio.
- Для загрузки данных используйте файлы формата CSV или Excel.

## Примечания
- Для работы Spark требуется установленная Java (устанавливается автоматически в Dockerfile).
- Все зависимости устанавливаются из `requirements.txt`.