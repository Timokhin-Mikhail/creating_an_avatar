# Описание проекта
Проект предназначен для создания аватара на основе сделанной в момент запуска фотографии.
Программа сделана как аналог программы от meta, видео-презентация которой находится в папке examples.
Также в папке examples имеется фотография, переданная для анализа, а также две обработанные фотографии,
выданные в качестве результата работы программы.


Сам проект состоит из одного файла main.py. Также для его работы необходима установка
Stable Diffusion web UI "AUTOMATIC1111" (https://github.com/AUTOMATIC1111/stable-diffusion-webui), а также установка связей, находящихся в файле requirements.txt.

## Порядок работы программы
После передачи в основную функцию full_work_with_photo промпта,параметров cfg scale и denoising strength происходит
включение видеокамеры с выводом снимаемого ей изображения на экран и запуском обратного отсчета до снимка экрана.
После этого в основную папку программы сохраняются два аватара с новым фоном и сохраненным изображением человека,
а также фото с измененным фоном и измененным изображением человека.


## Планируемые дополнения
1. Дополнение документации основного кода работы программы.
2. Создание web интерфейса, через который возможно будет вбивать данные для работы программы.