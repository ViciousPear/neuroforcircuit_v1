**Памятка по git:**
**Для bash команд необходимо открыть терминал: ctrl + ~**
Установка всех нужных расширений через requirements:
pip install -r requirements.txt. *для этого нужно скачать python 1.12.8

**Перед началом работы**:
Это необходимо сделать, чтобы перенести изменения из гита, если кто-то внес изменения
1) git switch название ветки - переход на свою ветку
2) git pull origin
3) git push origin название ветки - перенос изменений в свою ветку

**После окончания работы** (если были внесены изменения, которые были удачно внедрены)
1) git add . 
2) git commit -m "тут пишется комментарий что было внесено/изменено"
3) git push origin названиеветки
4) Зайти в github, создать pull request на нашу ветку. Это нужно, чтобы ветки не сломались